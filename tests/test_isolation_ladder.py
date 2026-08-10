"""Tests for eval/isolation_ladder.py.

The ladder's value is entirely in its discipline: the rungs must run in one
fixed order, it must stop at the first failure, and the verdict it prints must
be the one written down before the run rather than one composed afterwards to
fit the result. Those are the properties tested here.

Every test injects fakes for load / generate / synchronize / telemetry, so the
whole suite runs on a laptop with no torch, no CUDA, and no spend.
"""

from __future__ import annotations

import hashlib
import json
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
    step1 = rec.events[: rec.events.index("summary_written", rec.events.index("release"))]
    assert step1 == [
        "summary_written",           # rung marked 'running' before any work
        "load:1",
        "sync",                      # after load, before telemetry
        "telemetry:1:after_load",    # persisted BEFORE the dangerous call
        "enter_ctx:1",
        "generate:1",
        "exit_ctx:1",
        "sync",                      # after generate
        "telemetry:1:after_generate",  # carries the rung's true peak VRAM
        "release",
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


def test_the_summary_is_rewritten_at_the_start_and_end_of_every_rung() -> None:
    rec = Recorder(fail_at=3)
    rec.run()
    # 3 executed rungs x (start + end) = 6, plus one for the rung skipped after
    # the failure, so the durable artifact matches reality at every instant.
    assert len(rec.summaries) == 7
    assert rec.summaries[0] == ["running", "pending", "pending", "pending"]
    assert rec.summaries[1] == ["passed", "pending", "pending", "pending"]
    assert rec.summaries[-1] == ["passed", "passed", "failed", "skipped"]


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
# --- the partial-summary false-pass regression -------------------------------
def test_a_partial_summary_never_claims_all_four_passed() -> None:
    """The load-bearing regression.

    The summary is rewritten after every rung so it survives a hard death, which
    means "no rung has failed yet" is a state it is routinely written in. The
    previous version rendered that as ALL_PASS_VERDICT — literally "All four
    configurations generated cleanly" — so a process killed during rung 2 left a
    durable artifact claiming a clean sweep of rungs that never ran. A gate
    whose crash residue reads as a pass is worse than no gate.
    """
    results = [il.StepResult(step=step) for step in il.LADDER]
    results[0].status = "passed"
    results[1].status = "running"

    summary = il.summarise(results)
    assert summary["outcome"] == "incomplete"
    assert summary["complete"] is False if "complete" in summary else True
    assert "INCOMPLETE" in summary["verdict"]
    assert "NO CONCLUSION" in summary["verdict"]
    assert "All four configurations generated cleanly" not in summary["verdict"]
    assert il.ALL_PASS_VERDICT not in summary["verdict"]


def test_a_summary_written_before_any_rung_runs_is_incomplete() -> None:
    summary = il.summarise([il.StepResult(step=step) for step in il.LADDER])
    assert summary["outcome"] == "incomplete"
    assert summary["failed_at_step"] is None
    assert il.ALL_PASS_VERDICT not in summary["verdict"]


def test_the_running_rung_is_marked_so_a_hard_death_names_where_it_died() -> None:
    """Without this the last durable summary is the previous rung's, in which
    the current one is still 'pending' — indistinguishable from never attempted."""
    rec = Recorder()
    rec.run()
    # A summary is written at the START of each rung, before any work.
    assert rec.events.index("summary_written") < rec.events.index("load:1")
    assert rec.summaries[0][0] == "running"


def test_only_an_explicit_all_pass_is_green() -> None:
    """`failed_at_step is None` was the old green condition and it is true for an
    incomplete ladder too, which would have opened the launcher's gate on a run
    that never finished."""
    partial = [il.StepResult(step=step) for step in il.LADDER]
    partial[0].status = "passed"
    assert il.summarise(partial)["outcome"] != "all_passed"

    done = [il.StepResult(step=step) for step in il.LADDER]
    for result in done:
        result.status = "passed"
    assert il.summarise(done)["outcome"] == "all_passed"


# --- reproduction is a fault claim, not a rung index -------------------------
def test_a_non_cuda_failure_before_the_configuration_is_not_a_reproduction() -> None:
    """A gated-repo 401 or disk-full OSError at rung 1 fails before the probe's
    realized rung-3 configuration and is not the fault the probe died of."""
    results = [il.StepResult(step=step) for step in il.LADDER]
    results[0].status = "failed"
    results[0].error = "OSError: 401 Client Error: gated repo for meta-llama/Llama-3.1-8B-Instruct"

    summary = il.summarise(results)
    assert summary["failed_at_or_before_probe_configuration"] is True
    assert summary["failed_at_probe_configuration"] is False
    assert summary["fault_signature_matches_s0"] is False
    assert summary["s0_reproduction"] == "no_different_fault"


def test_the_actual_s0_fault_inside_the_configuration_is_a_reproduction() -> None:
    results = [il.StepResult(step=step) for step in il.LADDER]
    results[2].status = "failed"
    results[2].error = "RuntimeError: CUDA error: an illegal memory access was encountered"
    results[2].phase = "generate"

    summary = il.summarise(results)
    assert summary["fault_signature_matches_s0"] is True
    assert summary["failed_at_s0_phase"] is True
    assert summary["failed_at_probe_configuration"] is True
    assert summary["s0_reproduction"] == "yes_exact"


def test_same_signature_during_rung3_load_is_not_exact_reproduction() -> None:
    """The retained run completed load and adapter attachment before the first
    prompt faulted, so a load-phase illegal access is a different failure."""
    results = [il.StepResult(step=step) for step in il.LADDER]
    results[2].status = "failed"
    results[2].error = "RuntimeError: CUDA error: an illegal memory access was encountered"
    results[2].phase = "load"

    summary = il.summarise(results)
    assert summary["failed_at_probe_configuration"] is True
    assert summary["failed_at_s0_phase"] is False
    assert summary["fault_signature_matches_s0"] is True
    assert summary["s0_reproduction"] == "same_fault_at_probe_configuration_different_phase"


@pytest.mark.parametrize("fail_at", [1, 2])
def test_same_fault_before_probe_configuration_is_not_exact_reproduction(fail_at: int) -> None:
    results = [il.StepResult(step=step) for step in il.LADDER]
    results[fail_at - 1].status = "failed"
    results[fail_at - 1].error = "RuntimeError: CUDA error: an illegal memory access was encountered"

    summary = il.summarise(results)
    assert summary["failed_at_or_before_probe_configuration"] is True
    assert summary["failed_at_probe_configuration"] is False
    assert summary["fault_signature_matches_s0"] is True
    assert summary["s0_reproduction"] == "same_fault_before_probe_configuration"


def test_the_s0_fault_at_rung_4_is_outside_the_probes_configuration() -> None:
    """The probe never reached an adapter-enabled candidate, so rung 4 is new
    information rather than a reproduction — but the fault still matches."""
    results = [il.StepResult(step=step) for step in il.LADDER]
    for result in results[:3]:
        result.status = "passed"
    results[3].status = "failed"
    results[3].error = "RuntimeError: CUDA error: an illegal memory access was encountered"

    summary = il.summarise(results)
    assert summary["failed_at_or_before_probe_configuration"] is False
    assert summary["failed_at_probe_configuration"] is False
    assert summary["fault_signature_matches_s0"] is True
    assert summary["s0_reproduction"] == "same_fault_outside_the_probe_configuration"


@pytest.mark.parametrize(
    "error,expected",
    [
        ("RuntimeError: CUDA error: an illegal memory access was encountered", True),
        ("CUDA error: device-side assert triggered", False),
        ("OSError: [Errno 28] No space left on device", False),
        ("torch.cuda.OutOfMemoryError: CUDA out of memory", False),
        ("RuntimeError: NCCL operation failed", False),
        ("HTTPError: 401 Unauthorized", False),
        (None, False),
        ("", False),
    ],
)
def test_fault_signature_matching(error, expected) -> None:
    assert il.matches_s0_fault_signature(error) is expected


def test_summary_of_a_clean_ladder_does_not_claim_the_card_is_healthy() -> None:
    summary = il.summarise(Recorder().run())
    assert summary["outcome"] == "all_passed"
    assert summary["failed_at_step"] is None
    assert summary["s0_reproduction"] == "not_applicable_no_failure"
    assert "NOT REPRODUCED" in summary["verdict"]
    assert "does NOT clear the card" in summary["verdict"]


@pytest.mark.parametrize("fail_at,within", [(1, True), (2, True), (3, True), (4, False)])
def test_the_configuration_boundary_is_rung_3(fail_at: int, within: bool) -> None:
    """The probe crashed on candidate 'base' = PeftModel with the adapter
    disabled, which is rung 3. It never reached an adapter-enabled candidate, so
    rung 4 is outside the configuration it exercised. This is a fact about
    configurations, deliberately separate from any claim about the fault."""
    summary = il.summarise(Recorder(fail_at=fail_at).run())
    assert summary["failed_at_step"] == fail_at
    assert summary["failed_at_or_before_probe_configuration"] is within
    assert summary["failed_at_probe_configuration"] is (fail_at == 3)


def test_all_pass_verdict_does_not_blame_work_the_failed_probe_never_reached() -> None:
    summary = il.summarise(Recorder().run())
    verdict = summary["verdict"]
    assert "same first 610-token prompt" in verdict
    assert "before later prompts or categories" in verdict
    assert "prompt count (400 vs 1)" not in verdict
    assert "280-999" not in verdict
    assert "second category" not in verdict
    assert "thermal" not in verdict.lower()
    assert "long run" not in verdict.lower()
    assert "Intermittent or nondeterministic" in verdict


def test_the_old_conflated_field_is_gone() -> None:
    assert "reproduces_s0_probe_crash" not in il.summarise(Recorder(fail_at=2).run())


def test_summary_records_the_pinned_run_parameters() -> None:
    summary = il.summarise(Recorder().run())
    assert summary["expected_prompt_id"] == il.EXPECTED_PROMPT_ID
    assert summary["expected_prompt_sha256"] == il.EXPECTED_PROMPT_SHA256
    assert summary["expected_prompt_ids_sha256"] == il.EXPECTED_PROMPT_IDS_SHA256
    # The serialization travels with the digest; a bare hash is unverifiable.
    assert summary["prompt_ids_serialization"] == "utf-8 json, separators=(',',':')"
    assert summary["observed_prompt_tokens"] == il.OBSERVED_PROMPT_TOKENS
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


# --- prompt identity --------------------------------------------------------
class _FakeTokenizer:
    """Returns a fixed id sequence, so a test can pin what the digest sees."""

    def __init__(self, ids):
        self.ids = list(ids)

    def __call__(self, text):
        return {"input_ids": list(self.ids)}


def test_ids_digest_pins_its_serialization() -> None:
    """The encoding is part of the invariant: the same ids under json.dumps
    defaults produce a different digest, which is exactly the ambiguity that
    made 'hash the input ids' an underspecified instruction."""
    ids = [1, 2, 3]
    compact = il.prompt_ids_digest(ids)
    spaced = hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest()
    assert compact != spaced
    assert compact == hashlib.sha256(b"[1,2,3]").hexdigest()


def test_wrong_prompt_refuses_and_names_what_moved() -> None:
    """A pass on a different prompt is not evidence about the crash, so this is
    a refusal rather than a warning — and it must say whether the prompt string
    or its tokenization changed, since a count could not distinguish them."""
    with pytest.raises(ValueError) as excinfo:
        il.assert_prompt_identity(_FakeTokenizer([1, 2, 3]), "not the crash prompt")
    message = str(excinfo.value)
    assert "prompt sha256" in message
    assert "input-ids sha256" in message
    assert il.EXPECTED_PROMPT_SHA256 in message


def test_refusal_reports_token_count_as_informational_not_as_the_gate() -> None:
    with pytest.raises(ValueError) as excinfo:
        il.assert_prompt_identity(_FakeTokenizer([1, 2, 3]), "wrong")
    message = str(excinfo.value)
    assert "3 tokens" in message
    assert str(il.OBSERVED_PROMPT_TOKENS) in message
    assert "informational" in message


def test_mismatched_prompt_id_is_caught() -> None:
    with pytest.raises(ValueError) as excinfo:
        il.assert_prompt_identity(_FakeTokenizer([1, 2, 3]), "wrong", prompt_id="multiple_7")
    assert "prompt id" in str(excinfo.value)


def test_serialization_label_cannot_drift_from_the_separators() -> None:
    """The label is written into every artifact; if it were typed out by hand it
    could disagree with the encoding actually used, which is the same class of
    defect as an unnamed hash."""
    assert il.PROMPT_IDS_SERIALIZATION == "utf-8 json, separators=(',',':')"
    assert il.PROMPT_IDS_JSON_SEPARATORS == (",", ":")


def test_observed_identity_records_without_judging() -> None:
    """The refusal path needs the same structure as the success path, so this
    must return values for a prompt that will be rejected rather than raise."""
    observed = il.observed_prompt_identity(_FakeTokenizer([1, 2, 3]), "wrong", "multiple_7")
    assert observed["prompt_id"] == "multiple_7"
    assert observed["tokens"] == 3
    assert observed["ids_serialization"] == il.PROMPT_IDS_SERIALIZATION
    assert observed["prompt_sha256"] != il.EXPECTED_PROMPT_SHA256
    assert observed["prompt_ids_sha256"] == il.prompt_ids_digest([1, 2, 3])


def test_refusal_writes_a_self_contained_artifact_and_runs_no_rung(tmp_path, monkeypatch) -> None:
    """The critical path a manual check cannot protect.

    On 2026-08-09 the ladder refused and wrote nothing, so what the run saw
    survived only in stdout. This asserts the whole contract: exit 69, a durable
    v4 artifact carrying BOTH the pins and the observation, the provenance that
    makes an id hash meaningful (tokenizer and adapter revisions), and — the
    point of a smoke gate — that no rung executed.
    """
    monkeypatch.setenv(il._REEXEC_SENTINEL, "1")  # the ladder re-execs itself otherwise
    monkeypatch.setattr(il, "_load_tokenizer", lambda: _FakeTokenizer([7, 8, 9]))
    monkeypatch.setattr(
        il, "load_first_production_prompt", lambda tok, **kw: ("multiple_0", "not the crash prompt")
    )

    def _explode(*a, **k):  # pragma: no cover - fails the test if ever reached
        raise AssertionError("run_ladder must not execute after an identity refusal")

    monkeypatch.setattr(il, "run_ladder", _explode)

    rc = il.main(["--out-dir", str(tmp_path)])
    assert rc == il.EXIT_PROMPT_MISMATCH == 69

    written = json.loads((tmp_path / "isolation_ladder.json").read_text())
    assert written["schema"] == "isolation_ladder/v4"
    assert written["outcome"] == "refused_prompt_identity"
    assert written["complete"] is False

    # The pins, so the artifact says what was required...
    assert written["expected_prompt_sha256"] == il.EXPECTED_PROMPT_SHA256
    assert written["expected_prompt_ids_sha256"] == il.EXPECTED_PROMPT_IDS_SHA256
    # ...the observation, so it says what was actually seen...
    observed = written["observed_prompt_identity"]
    assert observed["verified"] is False
    assert observed["tokens"] == 3
    assert observed["prompt_sha256"] != il.EXPECTED_PROMPT_SHA256
    assert observed["ids_serialization"] == il.PROMPT_IDS_SERIALIZATION
    assert "prompt sha256" in observed["mismatch"]
    assert observed["mismatches"]
    # ...and the provenance, without which an id hash is not evidence.
    assert written["base_model"]["revision"] == il.BASE_MODEL_REVISION
    assert written["adapter"]["revision"] == il.SFT_ADAPTER_REVISION
    assert written["max_new_tokens"] == il.MAX_NEW_TOKENS
    assert "cuda_launch_blocking" in written
    assert all(s["status"] != "passed" for s in written["steps"])


def test_summary_schema_bumped_for_the_breaking_field_change() -> None:
    """v3 carried `expected_prompt_tokens`; v4 replaces it with identity fields.
    A reader keying on the old field must see a new schema, not a silent hole."""
    summary = il.summarise(Recorder().run())
    assert summary["schema"] == "isolation_ladder/v4"
    assert "expected_prompt_tokens" not in summary


def test_dry_run_needs_no_gpu_and_exits_clean() -> None:
    assert il.main(["--dry-run"]) == il.EXIT_OK


def test_no_current_fact_text_still_claims_the_retired_609_gate() -> None:
    """Regression for a defect that survived three review cycles of A1.

    A1 replaced the 609 token-count gate with a hash-identity gate, but five
    places kept asserting 609 as present-tense fact -- including inside
    ALL_PASS_VERDICT, so a green run would have printed a false statement about
    the crash it had just failed to reproduce. The explanatory comment that
    recounts the history is exempt; it is *about* the old value.
    """
    for path in (
        REPO_ROOT / "eval" / "isolation_ladder.py",
        REPO_ROOT / "docs" / "probe-bootstrap.md",
    ):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "609" not in line:
                continue
            # The history comment in isolation_ladder.py explains why 609 was
            # wrong; it must keep saying 609.
            assert line.lstrip().startswith("#"), (
                f"{path.name}:{lineno} states 609 as current fact: {line.strip()!r}"
            )


def test_all_pass_verdict_describes_the_prompt_the_gate_actually_pins() -> None:
    """The verdict is an artifact a human reads to decide what a green run
    means. It must describe the sequence the gate admitted -- 610 tokens with
    the duplicated BOS -- not the off-path count that never reached the model."""
    verdict = il.summarise(Recorder().run())["verdict"]
    assert str(il.OBSERVED_PROMPT_TOKENS) in verdict
    assert "609" not in verdict


def test_the_prompt_gate_cannot_be_overridden_from_the_command_line() -> None:
    """`--expect-prompt-tokens` let a caller bless any prompt while the docs
    promised a gate — a gate with a documented guarantee and a public override
    is not a gate. The gate is now identity (two hashes), not a count, but the
    override must stay absent for the same reason."""
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
