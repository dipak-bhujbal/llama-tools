"""`bos_handling` must describe what the code does, not what someone believed.

A provenance label is only worth the check behind it. `EXPECTED_PROMPT_TOKENS =
609` was a recorded belief with nothing tying it to the code, and it cost a pod
on 2026-08-09 before anyone noticed it had never been true of the production
path. These tests tie every field of BOS_HANDLING to observable behaviour, so
the manifest cannot drift into fiction the same way.

Two halves, deliberately kept apart:

* the STATIC claims -- what arguments the source actually passes -- checked by
  reading the call sites, no tokenizer required;
* the RESOLVED and OBSERVED behaviour -- what the pinned tokenizer really does
  with those arguments -- checked against the real tokenizer, skipped loudly
  when it is not in the local cache.

Scope is deliberate. BOS_HANDLING describes the EVAL generation path only.
`mining/backend.py` shares the construction and produced the pilot pairs, which
are study-2 training data; that is disclosed in ADR-009 rather than implied
here, and `applies_to` says so in the artifact itself.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval.bfcl_simple as bfcl  # noqa: E402


# --- static claims: does the source really pass these arguments? -------------
def test_render_arguments_match_the_recorded_claim() -> None:
    src = inspect.getsource(bfcl.build_prompt)
    assert ("tokenize=False" in src) is (bfcl.BOS_HANDLING["render_tokenize"] is False)
    assert ("add_generation_prompt=True" in src) is (
        bfcl.BOS_HANDLING["render_add_generation_prompt"] is True
    )


def test_retokenize_really_omits_add_special_tokens() -> None:
    """The recorded value is "omitted", so the call must not pass it. If someone
    later sets it explicitly, this fails rather than letting the label lie."""
    src = inspect.getsource(bfcl.generate)
    assert "tokenizer(prompt" in src
    assert "add_special_tokens" not in src
    assert bfcl.BOS_HANDLING["retokenize_add_special_tokens_argument"] == "omitted"


def test_manifest_carries_expectations_and_no_observation(tmp_path) -> None:
    """The pre-run manifest is written before the tokenizer runs, so it must not
    contain a field claiming something was observed."""
    manifest = bfcl.build_initial_manifest(
        _FakeArgs(tmp_path), ["base", "sft"], _real_category_paths(), n_prompts=3
    )
    recorded = manifest["bos_handling"]
    assert recorded == bfcl.BOS_HANDLING
    assert not any("observed" in key for key in recorded), (
        "an expectation must not be recorded under an 'observed' name before the "
        "tokenizer has run -- that is the 609 mistake"
    )
    assert recorded["applies_to"].startswith("eval generation path")


def test_manifest_object_is_copied_not_shared(tmp_path) -> None:
    """A run mutating its manifest must not rewrite the module constant."""
    manifest = bfcl.build_initial_manifest(
        _FakeArgs(tmp_path), ["base"], _real_category_paths(), n_prompts=1
    )
    manifest["bos_handling"]["choice_basis"] = "mutated"
    assert bfcl.BOS_HANDLING["choice_basis"] != "mutated"


def _real_category_paths():
    """Real pinned inputs: the manifest hashes them, so a stub would not do."""
    from bfcl_category_config import resolve_category_paths

    return resolve_category_paths(REPO_ROOT, "multiple")


class _FakeArgs:
    """Only the attributes build_initial_manifest reads."""

    def __init__(self, tmp_path: Path):
        self.category = "multiple"
        self.base_model = "meta-llama/Llama-3.1-8B-Instruct"
        self.base_revision = "rev"
        self.sft_adapter = "adapter"
        self.sft_adapter_subfolder = "adapter/"
        self.sft_adapter_revision = "rev"
        self.checkpoint_root = tmp_path
        self.checkpoints = []
        self.max_new_tokens = 512
        self.limit = None
        self.out_dir = tmp_path
        self.seed = 0


# --- resolved + observed behaviour: what the real tokenizer does -------------
@pytest.fixture(scope="module")
def pinned_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct", local_files_only=True
        )
    except Exception as exc:  # noqa: BLE001 - cold cache, gated repo, no auth
        pytest.skip(f"pinned tokenizer not in local cache: {type(exc).__name__}: {exc}")


def test_omitted_argument_really_resolves_to_true(pinned_tokenizer) -> None:
    """`retokenize_add_special_tokens_argument: "omitted"` only means double BOS
    because this tokenizer defaults it to True. If a future tokenizer revision
    defaulted it to False, the recorded expectation would silently be wrong."""
    observed = bfcl.observed_bos_handling(pinned_tokenizer, "hello")
    assert observed["resolved_add_special_tokens"] is (
        bfcl.BOS_HANDLING["expected_resolved_add_special_tokens"]
    )


def test_expected_leading_bos_count_is_what_actually_happens(pinned_tokenizer) -> None:
    from eval.isolation_ladder import load_first_production_prompt

    _, prompt = load_first_production_prompt(pinned_tokenizer)
    observed = bfcl.observed_bos_handling(pinned_tokenizer, prompt)
    assert (
        observed["observed_leading_bos_count"]
        == bfcl.BOS_HANDLING["expected_leading_bos_count"]
        == 2
    )


def test_observation_is_computed_not_copied_from_the_expectation(pinned_tokenizer) -> None:
    """A single-BOS input must report 1, or the observer is just echoing the pin."""
    rendered = pinned_tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False
    )
    stripped = rendered.replace(pinned_tokenizer.bos_token, "", 1)
    observed = bfcl.observed_bos_handling(pinned_tokenizer, stripped)
    assert observed["observed_leading_bos_count"] == 1


# --- lifecycle: does a RUN actually write the observation? -------------------
def test_the_runner_persists_the_observation_before_loading_a_model(
    pinned_tokenizer, tmp_path, monkeypatch
) -> None:
    """The gap the isolated helper tests could not see.

    `observed_bos_handling()` passing in isolation proves nothing about whether
    any real run calls it. Cycle-1 review found it was dead code: every manifest
    would have carried expectations only, while the source comment claimed a
    runtime observation. This drives the actual runner far enough to write the
    manifest, then reads that manifest off disk.

    Model loading is made to fail deliberately, because the observation is
    supposed to survive exactly that.
    """
    monkeypatch.setattr(
        bfcl.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: pinned_tokenizer)
    )

    def _no_model(*a, **k):
        raise RuntimeError("model loading deliberately fails after the observation")

    monkeypatch.setattr(bfcl.AutoModelForCausalLM, "from_pretrained", staticmethod(_no_model))
    monkeypatch.setattr(
        sys, "argv",
        ["bfcl_simple.py", "--category", "multiple", "--num-prompts", "2",
         "--sft-only", "--out-dir", str(tmp_path)],
    )

    with contextlib.suppress(BaseException):
        bfcl.main()

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert "bos_handling" in manifest, "expectations must still be recorded"
    observed = manifest.get("bos_handling_observed")
    assert observed is not None, (
        "a real run must persist the observation; if this fails, "
        "observed_bos_handling() has become dead code again"
    )
    assert observed["observed_leading_bos_count"] == 2
    assert observed["matches_expected_leading_bos_count"] is True
    assert observed["matches_expected_resolved_add_special_tokens"] is True
    assert observed["matches_approved_construction"] is True
    assert observed["measured_on"]["category"] == "multiple"
    assert observed["measured_on"]["prompt_id"]


class _SingleBosTokenizer:
    """Renders with a BOS but does NOT add another — i.e. a single-BOS
    construction, which is exactly what A2 option 1 did not approve."""

    pad_token = None
    eos_token = "<eos>"
    bos_token = "<bos>"
    bos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<bos>PROMPT"

    def __call__(self, text, add_special_tokens=True, **kwargs):
        ids = [self.bos_token_id] if text.startswith(self.bos_token) else []
        return {"input_ids": ids + [42, 43, 44]}


def test_a_mismatching_construction_refuses_before_the_model_loader(
    tmp_path, monkeypatch
) -> None:
    """Fail closed, not warn and continue.

    A mismatch means the run would measure a construction the owner did not
    approve. Continuing would spend on figures under an unapproved regime after
    a free preflight had already proved the run invalid. This asserts the model
    loader is never reached and the manifest records why.
    """
    monkeypatch.setattr(
        bfcl.AutoTokenizer, "from_pretrained",
        staticmethod(lambda *a, **k: _SingleBosTokenizer()),
    )

    reached = []

    def _loader(*a, **k):
        reached.append(True)
        raise AssertionError("model loader must not be reached after a BOS mismatch")

    monkeypatch.setattr(bfcl.AutoModelForCausalLM, "from_pretrained", staticmethod(_loader))
    monkeypatch.setattr(
        sys, "argv",
        ["bfcl_simple.py", "--category", "multiple", "--num-prompts", "2",
         "--sft-only", "--out-dir", str(tmp_path)],
    )

    with contextlib.suppress(BaseException):
        bfcl.main()

    assert not reached, "refusal must happen before any model is loaded"

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert "BosHandlingMismatch" in (manifest.get("failure_reason") or "")
    observed = manifest["bos_handling_observed"]
    assert observed["observed_leading_bos_count"] == 1
    assert observed["matches_approved_construction"] is False
