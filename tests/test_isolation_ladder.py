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

    def telemetry(self, model) -> dict:
        self.telemetry_models.append(model)
        self.events.append(f"telemetry:{str(model).split('-')[1]}")
        return {}

    def synchronize(self) -> None:
        self.events.append("sync")

    def release(self, model) -> None:
        self.released.append(model)
        self.events.append(f"release:{str(model).split('-')[1]}")

    def emit(self, line: str) -> None:
        self.lines.append(line)

    def run(self):
        return il.run_ladder(
            loaders=self.loaders(),
            generate_fn=self.generate,
            telemetry_fn=self.telemetry,
            synchronize_fn=self.synchronize,
            release_fn=self.release,
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
        "sync",          # after load, before telemetry: the fault must be attributed here
        "telemetry:1",
        "enter_ctx:1",
        "generate:1",
        "exit_ctx:1",
        "sync",          # after generate
        "release:1",
    ]


def test_telemetry_is_captured_for_every_executed_step() -> None:
    rec = Recorder()
    rec.run()
    assert rec.telemetry_models == ["model-1", "model-2", "model-3", "model-4"]


def test_every_loaded_model_is_released_so_vram_does_not_accumulate() -> None:
    """Without this a 16 GB model from step 2 is still resident while step 3
    loads another, and step 3 would OOM for reasons unrelated to the wrapper it
    is supposed to be testing."""
    rec = Recorder()
    rec.run()
    assert rec.released == ["model-1", "model-2", "model-3", "model-4"]


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
    assert rec.released == ["model-1", "model-2", "model-3"]


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
