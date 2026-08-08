"""Tests for eval/isolation_ladder.py.

The ladder's value is entirely in its discipline: the rungs must run in one
fixed order, it must stop at the first failure, and the verdict it prints must
be the one written down before the run rather than one composed afterwards to
fit the result. Those are the properties tested here.

Every test injects fakes for load / generate / synchronize / telemetry, so the
whole suite runs on a laptop with no torch, no CUDA, and no spend.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import isolation_ladder as il  # noqa: E402


class Recorder:
    """Collects every effect the ladder performs, in order."""

    def __init__(self, fail_at: int | None = None, fail_phase: str = "generate", exc=RuntimeError):
        self.fail_at = fail_at
        self.fail_phase = fail_phase
        self.exc = exc
        self.events: list[str] = []
        self.loaded: list[str] = []
        self.telemetry_models: list[object] = []
        self.released: list[object] = []
        self.persisted: list[str] = []
        self.summaries: list[list] = []
        self.lines: list[str] = []

    def _step_index_for(self, loader_key: str) -> int:
        return next(s.index for s in il.LADDER if s.loader == loader_key)

    def loaders(self) -> dict:
        def make(loader_key: str):
            def load():
                index = self._step_index_for(loader_key)
                self.loaded.append(loader_key)
                self.events.append(f"load:{index}")
                if index == self.fail_at and self.fail_phase == "load":
                    raise self.exc("CUDA error: an illegal memory access was encountered")
                model = f"model-{index}"

                class Ctx:
                    def __enter__(self_inner):
                        self.events.append(f"enter_ctx:{index}")
                        return self_inner

                    def __exit__(self_inner, *a):
                        self.events.append(f"exit_ctx:{index}")
                        return False

                return model, Ctx

            return load

        return {s.loader: make(s.loader) for s in il.LADDER}

    def generate(self, model) -> str:
        index = int(str(model).split("-")[1])
        self.events.append(f"generate:{index}")
        if index == self.fail_at and self.fail_phase == "generate":
            raise self.exc("CUDA error: an illegal memory access was encountered")
        return f"out-{index}"

    def snapshot(self, step, phase: str, model) -> dict:
        self.telemetry_models.append((step.index, phase, model))
        self.persisted.append(f"step{step.index}_{phase}")
        self.events.append(f"telemetry:{step.index}:{phase}")
        return {}

    def synchronize(self) -> None:
        self.events.append("sync")

    def release(self) -> None:
        self.released.append(True)
        self.events.append("release")

    def emit(self, line: str) -> None:
        self.lines.append(line)

    def progress(self, results) -> None:
        self.summaries.append([r.status for r in results])
        self.events.append("summary_written")

    def run(self):
        return il.run_ladder(
            loaders=self.loaders(),
            generate_fn=self.generate,
            snapshot_fn=self.snapshot,
            synchronize_fn=self.synchronize,
            release_fn=self.release,
            progress_fn=self.progress,
            emit=self.emit,
        )


# --- the rungs themselves ---------------------------------------------------
def test_ladder_has_exactly_the_four_specified_rungs_in_order() -> None:
    assert [s.index for s in il.LADDER] == [1, 2, 3, 4]
    assert [s.loader for s in il.LADDER] == [
        "raw_base_explicit_device",
        "raw_base_auto_device_map",
        "peft_adapter_disabled",
        "peft_adapter_enabled",
    ]


def test_every_declared_loader_actually_exists() -> None:
    """A rung naming a loader that isn't wired would abort mid-ladder on the pod,
    after the model was already downloaded and the meter already running."""
    for step in il.LADDER:
        assert step.loader in il.LOADERS, step.loader
        assert callable(il.LOADERS[step.loader])


def test_pinned_identities_match_the_miner_exactly() -> None:
    """The ladder restates the base/adapter constants instead of importing them.
    That is deliberate, and this test is the reason it is safe: a revision that
    drifts from mining/mine_pairs.py would mean the smoke gate exercises a
    different model than the run it is gating, which is the 'right number over
    the wrong population' defect in a new costume."""
    source = (REPO_ROOT / "mining" / "mine_pairs.py").read_text(encoding="utf-8")

    def literal(name: str) -> str:
        match = re.search(rf'^{name}\s*=\s*"([^"]+)"', source, re.MULTILINE)
        assert match, f"{name} not found in mining/mine_pairs.py"
        return match.group(1)

    assert il.BASE_MODEL_REPO == literal("BASE_MODEL_REPO")
    assert il.BASE_MODEL_REVISION == literal("BASE_MODEL_REVISION")
    assert il.SFT_ADAPTER_REPO == literal("SFT_ADAPTER_REPO")
    assert il.SFT_ADAPTER_SUBFOLDER == literal("SFT_ADAPTER_SUBFOLDER")
    assert il.SFT_ADAPTER_REVISION == literal("SFT_ADAPTER_REVISION")


def test_verdicts_are_distinct_and_name_their_suspect() -> None:
    verdicts = [s.verdict_on_failure for s in il.LADDER]
    assert len(set(verdicts)) == 4
    assert "PLACEMENT" in verdicts[1]
    assert "PEFT WRAPPER" in verdicts[2]
    assert "ADAPTER STATE" in verdicts[3]


# --- ordering, telemetry, synchronisation -----------------------------------
def test_all_pass_runs_every_rung_in_order() -> None:
    rec = Recorder()
    results = rec.run()
    assert [r.status for r in results] == ["passed"] * 4
    assert rec.loaded == [s.loader for s in il.LADDER]
    assert [r.generated_text for r in results] == ["out-1", "out-2", "out-3", "out-4"]


def test_synchronize_runs_after_load_and_after_generate_within_each_step() -> None:
    rec = Recorder()
    rec.run()
    step1 = rec.events[: rec.events.index("load:2")]
    assert step1 == [
        "load:1",
        "sync",                      # after load, before telemetry
        "telemetry:1:after_load",    # persisted BEFORE the dangerous call
        "enter_ctx:1",
        "generate:1",
        "exit_ctx:1",
        "sync",                      # after generate
        "telemetry:1:after_generate",  # carries the rung's true peak VRAM
        "release",
        "summary_written",
    ]


def test_telemetry_is_captured_before_and_after_the_operation_on_every_step() -> None:
    """One pre-operation sample is not enough: it cannot carry the rung's peak
    VRAM (generation had not run) and cannot see an ECC or Xid counter that the
    operation itself moved."""
    rec = Recorder()
    rec.run()
    assert rec.persisted == [
        f"step{i}_{phase}" for i in (1, 2, 3, 4) for phase in ("after_load", "after_generate")
    ]


def test_every_snapshot_is_persisted_as_it_is_taken_not_buffered() -> None:
    """Crash durability. The process being measured has already died abruptly
    once, taking all of its evidence with it; a snapshot that only reaches disk
    after the ladder returns is one the crashing case never produces."""
    rec = Recorder(fail_at=3, fail_phase="generate")
    rec.run()
    # Rungs 1 and 2 are fully on disk, and rung 3's pre-fault state survives.
    assert "step1_after_generate" in rec.persisted
    assert "step2_after_generate" in rec.persisted
    assert "step3_after_load" in rec.persisted
    assert "step3_on_failure" in rec.persisted


def test_the_summary_is_rewritten_after_every_rung(paths=None) -> None:
    rec = Recorder(fail_at=3)
    rec.run()
    # One write per executed rung, each a complete snapshot of progress so far.
    assert len(rec.summaries) == 3
    assert rec.summaries[0][0] == "passed"
    assert rec.summaries[-1][2] == "failed"


def test_post_fault_evidence_is_collected_on_the_failing_rung() -> None:
    """torch will usually be unusable on a poisoned CUDA context, but nvidia-smi
    and the kernel log are separate processes — the Xid code the driver just
    logged is the most diagnostic thing available and must still be read."""
    rec = Recorder(fail_at=2, fail_phase="generate")
    results = rec.run()
    assert "on_failure" in results[1].telemetry
    assert rec.events.count("telemetry:2:on_failure") == 1


def test_a_failing_telemetry_collector_cannot_mask_the_fault_it_was_measuring() -> None:
    rec = Recorder()

    def exploding_snapshot(step, phase, model):
        raise OSError("nvidia-smi segfaulted")

    results = il.run_ladder(
        loaders=rec.loaders(),
        generate_fn=rec.generate,
        snapshot_fn=exploding_snapshot,
        synchronize_fn=rec.synchronize,
        release_fn=rec.release,
        emit=rec.emit,
    )
    assert [r.status for r in results] == ["passed"] * 4
    assert results[0].telemetry["after_load"]["status"] == "unavailable"
    assert "nvidia-smi segfaulted" in results[0].telemetry["after_load"]["reason"]


def test_every_loaded_model_is_released_so_vram_does_not_accumulate() -> None:
    rec = Recorder()
    rec.run()
    assert len(rec.released) == 4


def test_the_model_is_unreachable_by_the_time_release_runs() -> None:
    """The load-bearing lifetime test.

    An earlier version called `release_fn(model)`, and `_real_release` did `del
    model` — which deleted only that function's own parameter. `run_ladder`
    still held the model, so its refcount never reached zero and
    `empty_cache()` freed nothing. Rung 3's 16 GB stayed resident while rung 4
    loaded another copy; on a 24 GB card rung 4 OOMs and the ladder reports
    ADAPTER STATE as the suspect. Not a crash — a confident wrong verdict.

    So this asserts the property directly: at the moment teardown runs, nothing
    anywhere still refers to the model.
    """
    import contextlib
    import gc
    import weakref

    class Model:
        pass

    refs: list = []
    alive_at_release: list[bool] = []

    def make_loader():
        def load():
            model = Model()
            refs.append(weakref.ref(model))
            # Rung 3 returns `model.disable_adapter`, a BOUND METHOD, which pins
            # the model exactly as firmly as the model variable does. Mimic that.
            return model, lambda: contextlib.nullcontext()

        return load

    def release():
        gc.collect()
        alive_at_release.append(any(ref() is not None for ref in refs))

    il.run_ladder(
        loaders={s.loader: make_loader() for s in il.LADDER},
        generate_fn=lambda model: "ok",
        snapshot_fn=lambda step, phase, model: {},
        synchronize_fn=lambda: None,
        release_fn=release,
        emit=lambda line: None,
    )

    assert len(alive_at_release) == 4
    assert alive_at_release == [False, False, False, False], (
        "a reference to the model was still live when teardown ran, so "
        "empty_cache() cannot reclaim its VRAM"
    )


def test_a_bound_method_context_factory_does_not_pin_the_model() -> None:
    """Rung 3's real loader returns `model.disable_adapter`. If run_ladder keeps
    that around, the model stays alive regardless of what it does with its own
    `model` variable."""
    import gc
    import weakref

    class Model:
        def ctx(self):
            import contextlib

            return contextlib.nullcontext()

    refs: list = []
    alive: list[bool] = []

    def load():
        model = Model()
        refs.append(weakref.ref(model))
        return model, model.ctx  # bound method, holds a strong ref to model

    def release():
        gc.collect()
        alive.append(any(ref() is not None for ref in refs))

    il.run_ladder(
        steps=il.LADDER[:1],
        loaders={il.LADDER[0].loader: load},
        generate_fn=lambda model: "ok",
        snapshot_fn=lambda step, phase, model: {},
        synchronize_fn=lambda: None,
        release_fn=release,
        emit=lambda line: None,
    )
    assert alive == [False]


# --- abort on first failure -------------------------------------------------
@pytest.mark.parametrize("fail_at", [1, 2, 3, 4])
def test_failure_aborts_and_later_rungs_are_recorded_as_skipped(fail_at: int) -> None:
    rec = Recorder(fail_at=fail_at)
    results = rec.run()
    statuses = [r.status for r in results]
    assert statuses[: fail_at - 1] == ["passed"] * (fail_at - 1)
    assert statuses[fail_at - 1] == "failed"
    assert statuses[fail_at:] == ["skipped"] * (4 - fail_at)
    # Skipped rungs are present with a reason, not omitted: a reader must be
    # able to tell "did not run" from "has no result".
    for result in results[fail_at:]:
        assert "earlier rung failed" in result.error


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_no_loader_after_the_failing_one_is_ever_called(fail_at: int) -> None:
    rec = Recorder(fail_at=fail_at)
    rec.run()
    assert rec.loaded == [s.loader for s in il.LADDER[:fail_at]]


def test_the_failing_step_prints_its_preregistered_verdict() -> None:
    rec = Recorder(fail_at=2)
    rec.run()
    output = "\n".join(rec.lines)
    assert "ISOLATION VERDICT — branch 2" in output
    assert il.LADDER[1].verdict_on_failure in output
    # And not somebody else's verdict.
    assert il.LADDER[2].verdict_on_failure not in output


def test_a_model_that_loaded_is_released_even_when_generation_fails() -> None:
    rec = Recorder(fail_at=3, fail_phase="generate")
    rec.run()
    assert len(rec.released) == 3


def test_failure_during_load_records_the_load_phase() -> None:
    rec = Recorder(fail_at=3, fail_phase="load")
    results = rec.run()
    assert results[2].phase == "load"
    assert il.summarise(results)["failed_phase"] == "load"


def test_failure_during_generate_records_the_generate_phase() -> None:
    rec = Recorder(fail_at=3, fail_phase="generate")
    results = rec.run()
    assert results[2].phase == "generate"


def test_non_exception_base_exceptions_are_caught_not_propagated() -> None:
    """A hard CUDA fault can surface as something outside the Exception
    hierarchy. Letting it escape would lose the ladder's verdict and its
    telemetry at the exact moment both become valuable."""

    class HardFault(BaseException):
        pass

    rec = Recorder(fail_at=2, exc=HardFault)
    results = rec.run()
    assert results[1].status == "failed"
    assert "HardFault" in results[1].error


def test_traceback_is_retained_for_the_failing_step() -> None:
    rec = Recorder(fail_at=2)
    results = rec.run()
    assert "Traceback" in results[1].traceback
    assert results[0].traceback is None


# --- summary / verdict selection --------------------------------------------
def test_summary_of_a_clean_ladder_does_not_claim_the_card_is_healthy() -> None:
    summary = il.summarise(Recorder().run())
    assert summary["failed_at_step"] is None
    assert summary["reproduces_s0_probe_crash"] is False
    assert "NOT REPRODUCED" in summary["verdict"]
    assert "does NOT clear the card" in summary["verdict"]


@pytest.mark.parametrize("fail_at,reproduced", [(1, True), (2, True), (3, True), (4, False)])
def test_only_failures_at_or_before_the_probes_own_configuration_count_as_a_reproduction(
    fail_at: int, reproduced: bool
) -> None:
    """The probe crashed on candidate 'base' = PeftModel with the adapter
    disabled, which is rung 3. It never reached an adapter-enabled candidate, so
    a rung-4-only failure is a new finding, not a reproduction."""
    summary = il.summarise(Recorder(fail_at=fail_at).run())
    assert summary["failed_at_step"] == fail_at
    assert summary["reproduces_s0_probe_crash"] is reproduced


def test_summary_records_the_pinned_run_parameters() -> None:
    summary = il.summarise(Recorder().run())
    assert summary["expected_prompt_tokens"] == 609
    assert summary["max_new_tokens"] == 8
    assert summary["category"] == "multiple"
    assert summary["base_model"]["revision"] == il.BASE_MODEL_REVISION
    assert summary["adapter"]["subfolder"] == "adapter/"
    assert len(summary["steps"]) == 4


def test_summary_is_json_serialisable() -> None:
    import json

    json.dumps(il.summarise(Recorder(fail_at=2).run()), default=str)


# --- CUDA_LAUNCH_BLOCKING ---------------------------------------------------
def test_reexec_is_needed_when_launch_blocking_is_absent_or_wrong() -> None:
    assert il.needs_launch_blocking_reexec({}) is True
    assert il.needs_launch_blocking_reexec({"CUDA_LAUNCH_BLOCKING": "0"}) is True
    assert il.needs_launch_blocking_reexec({"CUDA_LAUNCH_BLOCKING": ""}) is True


def test_reexec_is_not_needed_when_already_set() -> None:
    assert il.needs_launch_blocking_reexec({"CUDA_LAUNCH_BLOCKING": "1"}) is False


def test_reexec_never_loops_even_if_the_variable_did_not_take() -> None:
    """The sentinel is the loop guard. Without it, an environment where the
    variable cannot be set would fork-bomb the pod."""
    env = {il._REEXEC_SENTINEL: "1"}
    assert il.needs_launch_blocking_reexec(env) is False


def test_reexec_sets_both_the_variable_and_the_sentinel() -> None:
    captured: dict = {}

    def fake_execve(path, argv, env):
        captured["path"] = path
        captured["argv"] = argv
        captured["env"] = env

    il.reexec_with_launch_blocking(["ladder.py", "--out", "x.json"], {"PATH": "/bin"}, fake_execve)
    assert captured["env"]["CUDA_LAUNCH_BLOCKING"] == "1"
    assert captured["env"][il._REEXEC_SENTINEL] == "1"
    assert captured["env"]["PATH"] == "/bin"
    assert captured["argv"][1:] == ["ladder.py", "--out", "x.json"]


# --- prompt assertion -------------------------------------------------------
class _FakeTokenizer:
    def __init__(self, n: int):
        self.n = n

    def __call__(self, text):
        return {"input_ids": list(range(self.n))}


def test_prompt_token_count_must_match_exactly() -> None:
    assert il.assert_prompt_token_count(_FakeTokenizer(609), "prompt") == 609


def test_wrong_token_count_refuses_to_run_and_names_both_numbers() -> None:
    """A pass on a different prompt is not evidence about the crash, so this is
    a refusal rather than a warning."""
    with pytest.raises(ValueError) as excinfo:
        il.assert_prompt_token_count(_FakeTokenizer(512), "prompt")
    message = str(excinfo.value)
    assert "512" in message and "609" in message


def test_dry_run_needs_no_gpu_and_exits_clean() -> None:
    assert il.main(["--dry-run"]) == il.EXIT_OK


def test_the_609_token_invariant_cannot_be_overridden_from_the_command_line() -> None:
    """`--expect-prompt-tokens` let a caller bless any prompt length while the
    docs promised the gate refused anything but 609 — a gate with a documented
    guarantee and a public override is not a gate."""
    parser = il.build_parser()
    flags = {action.option_strings[0] for action in parser._actions if action.option_strings}
    assert "--expect-prompt-tokens" not in flags
    with pytest.raises(SystemExit):
        parser.parse_args(["--expect-prompt-tokens", "1"])


def test_a_real_run_requires_an_out_dir(capsys) -> None:
    """Evidence that can be lost is not evidence: the rung that kills the
    process is the one whose telemetry matters."""
    assert il.main([]) == il.EXIT_USAGE
    assert "--out-dir is required" in capsys.readouterr().err
