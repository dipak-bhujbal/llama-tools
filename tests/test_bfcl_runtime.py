"""Tests for the bfcl_simple.py run lifecycle: manifest start/finalize,
whole-directory refusal/overwrite, and on-disk output validation.

None of these tests load a real model, download weights, or touch the
network. AutoTokenizer, AutoModelForCausalLM, PeftModel, and generate() are
all replaced with fakes/monkeypatches, and the BFCL question/answer files are
tiny synthetic fixtures written to tmp_path — the real (large) eval/bfcl_data
files are never read. This is what makes it possible to exercise a wall-clock
"killed mid-run" scenario, the refuse-if-nonempty guard, and on-disk
validation without a GPU or a paid run.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import bfcl_simple
from bfcl_category_config import CategoryPaths

# ---------------------------------------------------------------------------
# Fakes: no model, tokenizer, or generate() call ever touches real weights.
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Records nothing; just enough surface for build_prompt() and the
    pad_token fallback in main() to run without a real tokenizer."""

    pad_token: str | None = None
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT::" + json.dumps(messages)


class FakeModel:
    """Records set_adapter()/disable_adapter()/load_adapter() calls so tests
    can assert on the "base never reaches set_adapter()" invariant, and on
    which adapters were attached, without a real PEFT model.
    """

    def __init__(self) -> None:
        self.device = "cpu"
        self.set_adapter_calls: list[str] = []
        self.disable_adapter_calls = 0
        self.load_adapter_calls: list[tuple[str, str]] = []
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def set_adapter(self, name: str) -> None:
        self.set_adapter_calls.append(name)

    @contextlib.contextmanager
    def disable_adapter(self):
        self.disable_adapter_calls += 1
        yield

    def load_adapter(self, path: str, adapter_name: str) -> None:
        self.load_adapter_calls.append((path, adapter_name))


@pytest.fixture
def fake_model_stack(monkeypatch):
    """Patch the model-loading surface with fakes. Returns a dict that gets
    populated with the created FakeModel under key "model" once main() has
    run, so tests can inspect its call-recording after the fact.
    """
    created: dict[str, FakeModel] = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeTokenizer()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            # Placeholder "base": PeftModel.from_pretrained below ignores it,
            # exactly like the real PeftModel wraps a real base model.
            return object()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base, adapter_path, adapter_name="sft", **kwargs):
            model = FakeModel()
            created["model"] = model
            return model

    monkeypatch.setattr(bfcl_simple, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(bfcl_simple, "AutoModelForCausalLM", FakeAutoModelForCausalLM)
    monkeypatch.setattr(bfcl_simple, "PeftModel", FakePeftModel)
    return created


def install_fake_generate(monkeypatch, *, fail_at: int | None = None,
                          sigterm_at: int | None = None):
    """Replace bfcl_simple.generate with a fake that never touches a model.

    fail_at: raise RuntimeError on the Nth call (1-indexed) — simulates a
    generic mid-run failure.
    sigterm_at: send this process a real SIGTERM on the Nth call — simulates
    a wall-clock timeout killing the run, exercising the actual signal
    handler installed by main() rather than a stand-in exception.
    """
    state = {"count": 0}

    def fake_generate(model, tokenizer, prompt, max_new_tokens):
        state["count"] += 1
        n = state["count"]
        if sigterm_at is not None and n == sigterm_at:
            os.kill(os.getpid(), signal.SIGTERM)
            # Reached only if the installed handler failed to interrupt
            # execution — fail loudly rather than silently completing the run.
            raise AssertionError("SIGTERM handler did not interrupt execution")
        if fail_at is not None and n == fail_at:
            raise RuntimeError(f"synthetic failure at call {n}")
        return json.dumps({"name": "irrelevant", "arguments": {}, "call": n})

    monkeypatch.setattr(bfcl_simple, "generate", fake_generate)
    return state


@pytest.fixture(autouse=True)
def _restore_sigterm_handler():
    """main() installs a process-global SIGTERM handler. Restore whatever was
    there before each test so a test that exercises SIGTERM can't leak a
    custom handler into unrelated tests later in the same pytest process.
    """
    original = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, original)


# ---------------------------------------------------------------------------
# Synthetic BFCL fixture: tiny, self-contained, never touches eval/bfcl_data.
# ---------------------------------------------------------------------------


def _write_bfcl_data(tmp_path: Path, ids: list[str]) -> tuple[Path, Path]:
    questions_path = tmp_path / "questions.jsonl"
    answer_path = tmp_path / "answers.jsonl"
    with questions_path.open("w") as f:
        for pid in ids:
            row = {
                "id": pid,
                "question": [[{"role": "user", "content": f"do the thing for {pid}"}]],
                "function": [
                    {
                        "name": "do_thing",
                        "description": "does the thing",
                        "parameters": {
                            "type": "dict",
                            "properties": {"x": {"type": "integer", "description": "x"}},
                            "required": ["x"],
                        },
                    }
                ],
            }
            f.write(json.dumps(row) + "\n")
    with answer_path.open("w") as f:
        for pid in ids:
            f.write(json.dumps({"id": pid, "ground_truth": [{"do_thing": {"x": [1]}}]}) + "\n")
    return questions_path, answer_path


def _patch_category_paths(monkeypatch, tmp_path: Path, ids: list[str]) -> Path:
    """Point resolve_category_paths at a synthetic fixture instead of the
    real (large) eval/bfcl_data files. Returns the (not-yet-created) out_dir.
    """
    questions_path, answer_path = _write_bfcl_data(tmp_path, ids)
    out_dir = tmp_path / "out"

    def fake_resolve(repo_root, category):
        return CategoryPaths(
            questions=questions_path, answer_key=answer_path, default_output=out_dir
        )

    monkeypatch.setattr(bfcl_simple, "resolve_category_paths", fake_resolve)
    return out_dir


def _base_argv(out_dir: Path, *extra: str) -> list[str]:
    return [
        "bfcl_simple.py",
        "--category", "simple_python",
        "--out-dir", str(out_dir),
        *extra,
    ]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_manifest(out_dir: Path) -> dict:
    return json.loads((out_dir / "run_manifest.json").read_text())


# ---------------------------------------------------------------------------
# 1. Fake end-to-end run: exact cartesian product of rows.
# ---------------------------------------------------------------------------


def test_fake_run_produces_exact_cartesian_product_of_rows(
    tmp_path, monkeypatch, fake_model_stack
):
    ids = [f"t{i}" for i in range(3)]
    out_dir = _patch_category_paths(monkeypatch, tmp_path, ids)
    ckpt_root = tmp_path / "ckpt_root"
    (ckpt_root / "checkpoint-1").mkdir(parents=True)
    (ckpt_root / "checkpoint-2").mkdir(parents=True)
    install_fake_generate(monkeypatch)

    argv = _base_argv(
        out_dir, "--include-base", "--checkpoints", "1", "2", "--checkpoint-root", str(ckpt_root)
    )
    monkeypatch.setattr(sys, "argv", argv)
    bfcl_simple.main()

    candidates = ["base", "sft", "dpo-1", "dpo-2"]
    rows = _read_jsonl(out_dir / "generations.jsonl")
    assert len(rows) == len(ids) * len(candidates)
    pairs = {(r["id"], r["model_name"]) for r in rows}
    assert pairs == {(pid, cand) for pid in ids for cand in candidates}


# ---------------------------------------------------------------------------
# 2. "base" is generated with adapters disabled, never via set_adapter().
# ---------------------------------------------------------------------------


def test_base_generated_with_adapters_disabled_and_never_set_adapter(
    tmp_path, monkeypatch, fake_model_stack
):
    ids = ["t0", "t1"]
    out_dir = _patch_category_paths(monkeypatch, tmp_path, ids)
    install_fake_generate(monkeypatch)

    argv = _base_argv(out_dir, "--include-base", "--sft-only")
    monkeypatch.setattr(sys, "argv", argv)
    bfcl_simple.main()

    model = fake_model_stack["model"]
    assert model.disable_adapter_calls == 1
    assert model.set_adapter_calls == ["sft"]
    assert "base" not in model.set_adapter_calls


# ---------------------------------------------------------------------------
# 3. A run interrupted partway: rows survive, manifest says incomplete.
# ---------------------------------------------------------------------------


def test_interrupted_run_leaves_partial_rows_and_incomplete_manifest(
    tmp_path, monkeypatch, fake_model_stack
):
    ids = [f"t{i}" for i in range(3)]
    out_dir = _patch_category_paths(monkeypatch, tmp_path, ids)
    install_fake_generate(monkeypatch, fail_at=2)  # succeeds once, then raises

    argv = _base_argv(out_dir, "--sft-only")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        bfcl_simple.main()

    rows = _read_jsonl(out_dir / "generations.jsonl")
    assert len(rows) == 1

    manifest = _read_manifest(out_dir)
    assert manifest["status"] == "incomplete"
    assert "RuntimeError" in manifest["failure_reason"]
    assert "synthetic failure" in manifest["failure_reason"]
    assert manifest["rows_written"] == 1
    assert "ended_utc" in manifest
    assert manifest["elapsed_seconds"] >= 0
    # Generations hash is recorded as-of-the-moment of failure.
    assert manifest["outputs"]["generations"]["sha256"] == hashlib.sha256(
        (out_dir / "generations.jsonl").read_bytes()
    ).hexdigest()
    # The run never got far enough to write a report.
    assert "report" not in manifest["outputs"]


def test_interrupted_run_via_real_sigterm_leaves_incomplete_manifest(
    tmp_path, monkeypatch, fake_model_stack
):
    """Sends this test process an actual SIGTERM mid-run (from inside the
    faked generate()) rather than simulating the failure with a plain
    exception, so the installed signal handler itself is exercised — not
    just the try/except/finally machinery downstream of it.
    """
    ids = [f"t{i}" for i in range(3)]
    out_dir = _patch_category_paths(monkeypatch, tmp_path, ids)
    install_fake_generate(monkeypatch, sigterm_at=2)

    argv = _base_argv(out_dir, "--sft-only")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(bfcl_simple._Terminated):
        bfcl_simple.main()

    rows = _read_jsonl(out_dir / "generations.jsonl")
    assert len(rows) == 1

    manifest = _read_manifest(out_dir)
    assert manifest["status"] == "incomplete"
    assert "signal" in manifest["failure_reason"].lower()
    assert manifest["rows_written"] == 1


# ---------------------------------------------------------------------------
# 4. A successful run finalizes to "complete" with hashes and timing.
# ---------------------------------------------------------------------------


def test_successful_run_finalizes_complete_with_hashes_and_timing(
    tmp_path, monkeypatch, fake_model_stack
):
    ids = ["t0", "t1"]
    out_dir = _patch_category_paths(monkeypatch, tmp_path, ids)
    install_fake_generate(monkeypatch)

    argv = _base_argv(out_dir, "--sft-only")
    monkeypatch.setattr(sys, "argv", argv)
    bfcl_simple.main()

    manifest = _read_manifest(out_dir)
    assert manifest["status"] == "complete"
    assert manifest["started_utc"]
    assert manifest["ended_utc"]
    assert manifest["elapsed_seconds"] >= 0
    assert manifest["rows_written"] == len(ids)  # 1 candidate ("sft") x 2 prompts
    assert manifest["outputs"]["generations"]["sha256"] == hashlib.sha256(
        (out_dir / "generations.jsonl").read_bytes()
    ).hexdigest()
    assert manifest["outputs"]["report"]["sha256"] == hashlib.sha256(
        (out_dir / "report.md").read_bytes()
    ).hexdigest()
    assert manifest["validation"]["ok"] is True
    assert "failure_reason" not in manifest
    # Package versions were recorded up front (item 1's environment block).
    pv = manifest["environment"]["package_versions"]
    for pkg in ("transformers", "peft", "accelerate", "torch", "huggingface_hub"):
        assert pkg in pv


# ---------------------------------------------------------------------------
# 5. Whole-directory refusal: fires independently for each sibling file.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stale_name", ["generations.jsonl", "report.md", "run_manifest.json"]
)
def test_refusal_fires_for_each_sibling_file_independently(
    tmp_path, monkeypatch, stale_name
):
    ids = ["t0"]
    out_dir = _patch_category_paths(monkeypatch, tmp_path, ids)
    out_dir.mkdir(parents=True)
    (out_dir / stale_name).write_text("stale non-empty content\n")

    argv = _base_argv(out_dir, "--sft-only")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        bfcl_simple.main()
    assert stale_name in str(exc_info.value)
    assert "refusing to write" in str(exc_info.value)

    # Refusal happens before the "running" manifest write — an empty stale
    # sibling must not be mistaken for evidence of a run that started.
    if stale_name != "run_manifest.json":
        assert not (out_dir / "run_manifest.json").exists()


# ---------------------------------------------------------------------------
# 6. --overwrite removes all three stale files before starting.
# ---------------------------------------------------------------------------


def test_overwrite_removes_all_three_stale_files(tmp_path, monkeypatch, fake_model_stack):
    ids = ["t0"]
    out_dir = _patch_category_paths(monkeypatch, tmp_path, ids)
    out_dir.mkdir(parents=True)
    for name in bfcl_simple.SIBLING_FILENAMES:
        (out_dir / name).write_text("stale\n")
    install_fake_generate(monkeypatch)

    argv = _base_argv(out_dir, "--sft-only", "--overwrite")
    monkeypatch.setattr(sys, "argv", argv)
    bfcl_simple.main()  # must not raise the refusal SystemExit

    manifest = _read_manifest(out_dir)
    assert manifest["status"] == "complete"
    assert "stale" not in (out_dir / "generations.jsonl").read_text()
    assert "stale" not in (out_dir / "report.md").read_text()


# ---------------------------------------------------------------------------
# 7. validate_run_outputs: rejects a short file, a duplicate, a missing pair.
# ---------------------------------------------------------------------------


def test_validate_run_outputs_accepts_well_formed_file(tmp_path):
    ids = ["a", "b"]
    candidates = ["sft", "dpo-1"]
    gen_path = tmp_path / "generations.jsonl"
    with gen_path.open("w") as f:
        for pid in ids:
            for cand in candidates:
                f.write(json.dumps({"id": pid, "model_name": cand}) + "\n")

    result = bfcl_simple.validate_run_outputs(tmp_path, ids, candidates)
    assert result["ok"] is True
    assert result["rows_on_disk"] == 4
    assert result["expected_rows"] == 4
    assert result["duplicate_pairs"] == []
    assert result["missing_pairs"] == []
    assert result["extra_pairs"] == []


def test_validate_run_outputs_rejects_short_file(tmp_path):
    """Total row count below expected — the straightforward "run got cut
    off partway" case."""
    ids = ["a", "b"]
    candidates = ["sft"]
    gen_path = tmp_path / "generations.jsonl"
    gen_path.write_text(json.dumps({"id": "a", "model_name": "sft"}) + "\n")

    result = bfcl_simple.validate_run_outputs(tmp_path, ids, candidates)
    assert result["ok"] is False
    assert result["rows_on_disk"] == 1
    assert result["expected_rows"] == 2
    assert ("b", "sft") in result["missing_pairs"]


def test_validate_run_outputs_rejects_duplicated_pair(tmp_path):
    ids = ["a", "b"]
    candidates = ["sft"]
    gen_path = tmp_path / "generations.jsonl"
    with gen_path.open("w") as f:
        f.write(json.dumps({"id": "a", "model_name": "sft"}) + "\n")
        f.write(json.dumps({"id": "a", "model_name": "sft"}) + "\n")  # duplicate
        f.write(json.dumps({"id": "b", "model_name": "sft"}) + "\n")

    result = bfcl_simple.validate_run_outputs(tmp_path, ids, candidates)
    assert result["ok"] is False
    assert ("a", "sft") in result["duplicate_pairs"]


def test_validate_run_outputs_rejects_missing_pair_even_when_row_count_matches(tmp_path):
    """A duplicate can mask a missing pair while keeping the total row count
    correct — exactly why missing/duplicate are checked as independent set
    operations rather than folded into a single count comparison."""
    ids = ["a", "b", "c"]
    candidates = ["sft"]
    gen_path = tmp_path / "generations.jsonl"
    with gen_path.open("w") as f:
        f.write(json.dumps({"id": "a", "model_name": "sft"}) + "\n")
        f.write(json.dumps({"id": "a", "model_name": "sft"}) + "\n")  # dup masks c missing
        f.write(json.dumps({"id": "b", "model_name": "sft"}) + "\n")

    result = bfcl_simple.validate_run_outputs(tmp_path, ids, candidates)
    assert result["rows_on_disk"] == result["expected_rows"] == 3
    assert result["ok"] is False
    assert ("c", "sft") in result["missing_pairs"]
    assert ("a", "sft") in result["duplicate_pairs"]


def test_validate_run_outputs_rejects_unparseable_line(tmp_path):
    ids = ["a"]
    candidates = ["sft"]
    gen_path = tmp_path / "generations.jsonl"
    gen_path.write_text("{not valid json\n")

    result = bfcl_simple.validate_run_outputs(tmp_path, ids, candidates)
    assert result["ok"] is False
    assert result["unparseable_lines"] == 1


def test_manifest_write_failure_does_not_mask_the_real_error(tmp_path, monkeypatch, capsys):
    """An exception raised in a finally block replaces the one already
    propagating. If writing the manifest fails, the reason the run actually
    died must still reach the operator."""
    import bfcl_simple

    def exploding_write(out_dir, manifest):
        raise OSError("disk full")

    monkeypatch.setattr(bfcl_simple, "write_run_manifest", exploding_write)

    manifest: dict = {}
    gen_path = tmp_path / "generations.jsonl"
    report_path = tmp_path / "report.md"

    # Reproduce the finally-block contract directly: the primary exception
    # must survive, and the secondary failure must be reported, not swallowed.
    primary = RuntimeError("the real failure")
    try:
        try:
            raise primary
        finally:
            try:
                bfcl_simple._finalize_manifest(
                    manifest, status="incomplete", started_monotonic=0.0,
                    rows_written=3, gen_path=gen_path, report_path=report_path,
                    failure_reason="RuntimeError: the real failure", validation=None,
                )
                bfcl_simple.write_run_manifest(tmp_path, manifest)
            except Exception as exc:
                print(f"WARNING: failed to write run manifest: {exc}")
    except RuntimeError as surfaced:
        assert surfaced is primary, "the primary exception was masked"
    else:
        raise AssertionError("primary exception did not propagate")

    assert "disk full" in capsys.readouterr().out
    assert manifest["status"] == "incomplete"
    assert manifest["rows_written"] == 3


# ---------------------------------------------------------------------------
# 12. Development subset: main() restricts BOTH files before reconciling ids.
#
# This is the integration the gate's own tests never covered. The first version
# of the scorer filtered the questions and left the answer key at parent size,
# so main() aborted on its id-equality check and no development run could ever
# complete. Testing the gate in isolation could not see that; only driving
# main() can.
# ---------------------------------------------------------------------------


def _patch_dev_category_paths(monkeypatch, tmp_path: Path, *, pinned, extras) -> Path:
    """A synthetic `live_multiple` parent = pinned ids + extras that must never
    be scored, with the pin loader returning only `pinned`."""
    questions_path, answer_path = _write_bfcl_data(tmp_path, [*pinned, *extras])
    out_dir = tmp_path / "out"

    def fake_resolve(repo_root, category):
        return CategoryPaths(
            questions=questions_path,
            answer_key=answer_path,
            default_output=out_dir,
            is_development_subset=True,
        )

    monkeypatch.setattr(bfcl_simple, "resolve_category_paths", fake_resolve)
    monkeypatch.setattr(bfcl_simple, "load_dev_subset_ids", lambda repo_root: list(pinned))
    return out_dir


def test_development_run_completes_and_scores_only_the_pinned_ids(
    tmp_path, monkeypatch, fake_model_stack
):
    pinned = [f"live_multiple_{i}" for i in range(3)]
    # An excluded collision id and a plain extra: both are in the parent files
    # and neither may reach a generation row.
    extras = ["live_multiple_190-84-0", "live_multiple_spare"]
    out_dir = _patch_dev_category_paths(monkeypatch, tmp_path, pinned=pinned, extras=extras)
    install_fake_generate(monkeypatch)

    # --sft-only is the §3.3 shape: shipped SFT scored once on D.
    argv = [
        "bfcl_simple.py",
        "--category", "live_multiple",
        "--out-dir", str(out_dir),
        "--sft-only",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    # Against the questions-only version this raises ValueError on the id
    # reconciliation, because the answer key still holds every parent row.
    bfcl_simple.main()

    rows = _read_jsonl(out_dir / "generations.jsonl")
    assert {r["id"] for r in rows} == set(pinned)
    assert len(rows) == len(pinned)
    for excluded in extras:
        assert all(r["id"] != excluded for r in rows)

    manifest = _read_manifest(out_dir)
    assert manifest["status"] == "complete"


def test_development_run_refuses_a_partial_run_via_num_prompts(
    tmp_path, monkeypatch, fake_model_stack
):
    """A truncated development run is not comparable to the frozen baseline, so
    the flag combination is refused before any model loads."""
    pinned = [f"live_multiple_{i}" for i in range(3)]
    out_dir = _patch_dev_category_paths(monkeypatch, tmp_path, pinned=pinned, extras=[])
    install_fake_generate(monkeypatch)

    argv = [
        "bfcl_simple.py",
        "--category", "live_multiple",
        "--out-dir", str(out_dir),
        "--sft-only",
        "--num-prompts", "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="num-prompts"):
        bfcl_simple.main()
