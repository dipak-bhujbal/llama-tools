"""§0 smoke gate, built as an isolation ladder.

The 2026-08-08 §0 probe died at 64s with `CUDA error: an illegal memory access
was encountered`, having written zero generations. Re-running it would cost the
same money to learn the same nothing, because the probe varies four things at
once: device placement, the PEFT wrapper, adapter state, and scale. A crash tells
you only that some combination is bad.

This ladder varies exactly one thing per rung, in an order chosen so each
adjacent pair isolates a single suspect:

    Step 1  raw base, explicit cuda:0, no device_map
    Step 2  raw base, device_map="auto"            -> 1 vs 2 isolates PLACEMENT
    Step 3  PeftModel, adapter DISABLED            -> 2 vs 3 isolates the WRAPPER
    Step 4  PeftModel, adapter ENABLED             -> 3 vs 4 isolates ADAPTER STATE

Step 3 is the configuration the probe's first candidate actually ran
(`bfcl_simple.py:570`, `model.disable_adapter()`), and step 4 is the
configuration the mining pilot ran successfully on 2026-08-07. So the ladder is
bracketed by one known failure and one known success.

Three properties make the result interpretable rather than suggestive:

* `CUDA_LAUNCH_BLOCKING=1`, set before torch is imported (this module re-execs
  itself to guarantee that). Without it a CUDA fault surfaces at an arbitrary
  later synchronisation point and the step it is attributed to is not the step
  that caused it — which is precisely why the postmortem could not name a cause.
* `torch.cuda.synchronize()` after load and after generate, within every step,
  so a fault cannot drift across a step boundary.
* Abort on first failure. Continuing past a CUDA illegal access is meaningless;
  the context is already poisoned and every later step would fail for a reason
  that has nothing to do with what it was meant to test.

Cost: one 609-token prompt and 8 new tokens per step. Seconds of GPU time.

`torch`, `transformers` and `peft` are imported lazily inside the loaders so
this module, its ordering, its verdict logic and its tests all run on a laptop
with no CUDA at zero cost.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import gpu_telemetry  # noqa: E402
from eval.bfcl_category_config import resolve_category_paths  # noqa: E402

# Exit codes continue launch_probe.sh's contract (0/64..68) rather than starting
# a second numbering scheme, so an operator reading either one reads the same
# table.
EXIT_OK = 0
EXIT_USAGE = 64
EXIT_PROMPT_MISMATCH = 69
EXIT_LADDER_FAILED = 70

# Pinned identities, byte-identical to mining/mine_pairs.py. They are restated
# rather than imported because importing that module pulls in the miner's own
# dependency stack, and the smoke gate must be startable when that stack is the
# thing under suspicion. Any drift between the two is caught by a test.
BASE_MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
SFT_ADAPTER_REPO = "centuriandip/llama-3.1-8b-tools-sft"
SFT_ADAPTER_SUBFOLDER = "adapter/"
SFT_ADAPTER_REVISION = "b6f4da479f8c6fc044ee8b802a92f47780f970c5"

# The probe's first paid generation was category=multiple, and codex measured its
# first prompt at 609 tokens on the pinned tokenizer. The ladder asserts this
# rather than logging it: if the prompt is not that prompt, the ladder is not
# exercising the thing that crashed, and a green result would be worthless.
PROBE_CATEGORY = "multiple"
EXPECTED_PROMPT_TOKENS = 609
MAX_NEW_TOKENS = 8

_REEXEC_SENTINEL = "_LADDER_LAUNCH_BLOCKING_REEXEC"


@dataclass(frozen=True)
class LadderStep:
    index: int
    name: str
    loader: str
    # What a failure at this rung implicates, given that every earlier rung
    # passed. Written out here so the verdict is fixed before the run, not
    # composed afterwards to fit whatever happened (Ground Rule 8 in spirit:
    # the interpretation is preregistered alongside the test).
    verdict_on_failure: str


LADDER: tuple[LadderStep, ...] = (
    LadderStep(
        index=1,
        name="raw base, explicit cuda:0, no device_map",
        loader="raw_base_explicit_device",
        verdict_on_failure=(
            "The simplest possible path already faults: unwrapped base weights, "
            "one explicit device, no accelerate dispatch. Placement, the PEFT "
            "wrapper and adapter state are all excluded because none of them is "
            "present. Implicates the card, the driver, the torch/CUDA build, or "
            "the base weights at this revision."
        ),
    ),
    LadderStep(
        index=2,
        name='raw base, device_map="auto"',
        loader="raw_base_auto_device_map",
        verdict_on_failure=(
            "PLACEMENT. The only difference from step 1 is device_map='auto', "
            "i.e. accelerate's dispatch and any CPU/disk offload it chose. PEFT "
            "is not involved and no adapter is loaded. Check the recorded "
            "hf_device_map for modules that landed off the GPU."
        ),
    ),
    LadderStep(
        index=3,
        name="PeftModel, adapter DISABLED (the probe's base candidate)",
        loader="peft_adapter_disabled",
        verdict_on_failure=(
            "THE PEFT WRAPPER. Step 2 exonerates placement; the only new element "
            "is PeftModel.from_pretrained plus the disable_adapter() context. "
            "This is exactly what bfcl_simple.py runs for candidate 'base', so a "
            "failure here reproduces the §0 probe crash and localises it to the "
            "wrapper rather than to the adapter weights."
        ),
    ),
    LadderStep(
        index=4,
        name="PeftModel, adapter ENABLED (what the mining pilot ran)",
        loader="peft_adapter_enabled",
        verdict_on_failure=(
            "ADAPTER STATE. Step 3 exonerates the wrapper itself; only the active "
            "LoRA weights differ. Note the mining pilot ran this same "
            "configuration successfully on 2026-08-07, so a failure here is a "
            "regression since that date — environment, card, or adapter revision "
            "— not a standing defect in the code path."
        ),
    ),
)

INCOMPLETE_VERDICT = (
    "INCOMPLETE — NO CONCLUSION. Not every rung ran to completion, and none has "
    "failed. This is a partial record written mid-run, not a result: if you are "
    "reading it after the process died, the rungs still marked 'pending' were "
    "never attempted and the one marked 'running' is where it died. Do NOT read "
    "the absence of a failed rung as a pass. Check the per-rung telemetry "
    "snapshots and the launcher's exit record for what actually happened."
)

ALL_PASS_VERDICT = (
    "NOT REPRODUCED. All four configurations generated cleanly. The ladder "
    "excludes placement, the PEFT wrapper and adapter state as *sufficient* "
    "causes at this scale, and points at something it deliberately does not "
    "vary: prompt count (400 vs 1), sequence length across the full 280-999 "
    "range, the second category, or time/thermal effects during a long run. It "
    "does NOT clear the card — a green ladder on a sick GPU is possible."
)


@dataclass
class StepResult:
    step: LadderStep
    status: str = "pending"  # pending | running | passed | failed | skipped
    phase: str = ""  # load | synchronize_after_load | generate | synchronize_after_generate
    seconds: float = 0.0
    generated_text: str | None = None
    error: str | None = None
    traceback: str | None = None
    telemetry: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "index": self.step.index,
            "name": self.step.name,
            "loader": self.step.loader,
            "status": self.status,
            "failed_phase": self.phase if self.status == "failed" else None,
            "seconds": round(self.seconds, 3),
            "generated_text": self.generated_text,
            "error": self.error,
            "traceback": self.traceback,
            "telemetry": self.telemetry,
        }


# ---------------------------------------------------------------------------
# CUDA_LAUNCH_BLOCKING has to be in the environment before the CUDA context is
# created. Setting it inside a running process is a no-op that looks like it
# worked, so the module re-execs itself instead of trusting os.environ.
# ---------------------------------------------------------------------------
def needs_launch_blocking_reexec(environ: dict) -> bool:
    if environ.get(_REEXEC_SENTINEL) == "1":
        return False  # already re-exec'd once; never loop
    return environ.get("CUDA_LAUNCH_BLOCKING") != "1"


def reexec_with_launch_blocking(argv: list[str], environ: dict, execve=os.execve) -> None:
    new_env = dict(environ)
    new_env["CUDA_LAUNCH_BLOCKING"] = "1"
    new_env[_REEXEC_SENTINEL] = "1"
    print(
        "Re-exec with CUDA_LAUNCH_BLOCKING=1 (it must precede CUDA context "
        "creation; setting it in-process would silently do nothing).",
        flush=True,
    )
    execve(sys.executable, [sys.executable, *argv], new_env)


# ---------------------------------------------------------------------------
# Prompt construction: the production function, on the production data, with the
# token count enforced.
# ---------------------------------------------------------------------------
def load_first_production_prompt(tokenizer, repo_root: Path = REPO_ROOT, category: str = PROBE_CATEGORY):
    """Build the first prompt of the probe's first paid category.

    `build_prompt` is imported from bfcl_simple rather than reimplemented. A
    reimplementation could differ from production in exactly the detail that
    matters and would then prove nothing about production.
    """
    from eval.bfcl_simple import build_prompt, load_jsonl

    paths = resolve_category_paths(repo_root, category)
    rows = load_jsonl(paths.questions)
    if not rows:
        raise ValueError(f"no prompts in {paths.questions}")
    first = rows[0]
    return first["id"], build_prompt(tokenizer, first["question"], first["function"])


def assert_prompt_token_count(tokenizer, prompt: str, expected: int = EXPECTED_PROMPT_TOKENS) -> int:
    actual = len(tokenizer(prompt)["input_ids"])
    if actual != expected:
        raise ValueError(
            f"prompt is {actual} tokens, expected {expected}. The ladder refuses "
            f"to run: a different prompt is not the prompt that crashed, and a "
            f"pass on it would not be evidence about the §0 failure."
        )
    return actual


# ---------------------------------------------------------------------------
# The four loaders. Each returns (model, generate_context_factory).
# ---------------------------------------------------------------------------
def _load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_REPO, revision=BASE_MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_raw_base(device_map):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "revision": BASE_MODEL_REVISION,
        "dtype": torch.bfloat16,  # matches bfcl_simple.py:528 exactly
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_REPO, **kwargs)
    if device_map is None:
        # Explicit single-device placement, the thing step 1 is testing. Done
        # after load so accelerate never sees a device_map at all.
        model = model.to("cuda:0")
    model.eval()
    return model


def _wrap_with_peft(base):
    from peft import PeftModel

    return PeftModel.from_pretrained(
        base,
        SFT_ADAPTER_REPO,
        adapter_name="sft",
        subfolder=SFT_ADAPTER_SUBFOLDER.rstrip("/"),
        revision=SFT_ADAPTER_REVISION,
    )


def loader_raw_base_explicit_device():
    import contextlib

    return _load_raw_base(device_map=None), contextlib.nullcontext


def loader_raw_base_auto_device_map():
    import contextlib

    return _load_raw_base(device_map="auto"), contextlib.nullcontext


def loader_peft_adapter_disabled():
    # device_map="auto" here, not cuda:0: step 3 must differ from step 2 in the
    # wrapper ALONE. Changing placement at the same time would collapse two
    # variables into one rung and destroy the isolation the ladder exists for.
    model = _wrap_with_peft(_load_raw_base(device_map="auto"))
    model.eval()
    # disable_adapter() is a context manager, entered around generation exactly
    # as bfcl_simple.py:570-577 does for candidate "base".
    return model, model.disable_adapter


def loader_peft_adapter_enabled():
    import contextlib

    model = _wrap_with_peft(_load_raw_base(device_map="auto"))
    model.set_adapter("sft")
    model.eval()
    return model, contextlib.nullcontext


LOADERS: dict[str, Callable[[], tuple[Any, Callable[[], Any]]]] = {
    "raw_base_explicit_device": loader_raw_base_explicit_device,
    "raw_base_auto_device_map": loader_raw_base_auto_device_map,
    "peft_adapter_disabled": loader_peft_adapter_disabled,
    "peft_adapter_enabled": loader_peft_adapter_enabled,
}


# ---------------------------------------------------------------------------
# Orchestration. Every effect is injected so the ordering, abort behaviour and
# verdict selection are testable with fakes and no GPU.
# ---------------------------------------------------------------------------
def run_ladder(
    *,
    steps: tuple[LadderStep, ...] = LADDER,
    loaders: dict[str, Callable[[], tuple[Any, Callable[[], Any]]]] | None = None,
    generate_fn: Callable[[Any], str],
    snapshot_fn: Callable[[LadderStep, str, Any], dict],
    synchronize_fn: Callable[[], None],
    release_fn: Callable[[], None],
    reset_peak_fn: Callable[[], None] = lambda: None,
    progress_fn: Callable[[list["StepResult"]], None] = lambda results: None,
    emit: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
) -> list[StepResult]:
    """Run the rungs in order, aborting at the first failure.

    Returns one StepResult per step; steps after a failure are 'skipped', which
    is recorded explicitly rather than omitted — a reader must be able to tell
    "step 4 was not run" from "step 4 has no result".

    `snapshot_fn(step, phase, model)` must both collect telemetry AND persist it
    before returning. Every call site below is placed on the assumption that the
    process may not survive the next line: the process being measured is one
    that has already died once, abruptly, taking all of its evidence with it.
    Telemetry that only reaches disk after the ladder returns is telemetry the
    crashing case never produces.

    `release_fn` takes no arguments on purpose. Passing the model to a teardown
    function binds it into that function's frame, so the refcount never reaches
    zero and `empty_cache()` frees nothing — the previous rung's 16 GB stays
    resident while the next rung loads another copy, and rung 4 OOMs on a 24 GB
    card for reasons that have nothing to do with adapter state. That would not
    look like a bug; it would look like a verdict.
    """
    loaders = LOADERS if loaders is None else loaders
    results = [StepResult(step=s) for s in steps]
    aborted = False

    def snapshot(result: StepResult, phase: str, model: Any) -> None:
        """Collect + persist, and never let a telemetry failure mask the fault
        that made the telemetry interesting."""
        try:
            bundle = snapshot_fn(result.step, phase, model)
        except BaseException as exc:  # noqa: BLE001
            result.telemetry[phase] = gpu_telemetry.unavailable(
                f"telemetry collection itself failed: {type(exc).__name__}: {exc}"
            )
            emit(f"  (telemetry '{phase}' could not be collected: {exc!r})")
            return
        result.telemetry[phase] = bundle
        gaps = gpu_telemetry.unavailable_fields(bundle)
        if gaps:
            emit(f"  telemetry gaps at '{phase}' ({len(gaps)}) — unmeasured, not zero:")
            for gap in gaps:
                emit(f"    - {gap}")

    for result in results:
        step = result.step
        if aborted:
            result.status = "skipped"
            result.error = "not run: an earlier rung failed and the CUDA context is unusable"
            try:
                progress_fn(results)
            except Exception as exc:  # pragma: no cover
                emit(f"  (progress write for skipped step {step.index} raised {exc!r})")
            continue

        emit("")
        emit("=" * 72)
        emit(f"STEP {step.index}/{len(steps)}: {step.name}")
        emit("=" * 72)

        # Mark and persist BEFORE doing anything, so a hard death leaves an
        # artifact that names the rung it died on. Without this the last
        # durable summary is the previous rung's, in which this one is still
        # 'pending' — indistinguishable from never having been attempted.
        result.status = "running"
        try:
            progress_fn(results)
        except Exception as exc:  # pragma: no cover
            emit(f"  (progress write before step {step.index} raised {exc!r}; continuing)")

        reset_peak_fn()
        started = clock()
        model = None
        gen_context_factory = None
        try:
            result.phase = "load"
            model, gen_context_factory = loaders[step.loader]()

            result.phase = "synchronize_after_load"
            synchronize_fn()

            # After load, before the dangerous call. This is the last moment the
            # model provably exists and placement is resolved, so it is written
            # to disk here rather than accumulated in memory.
            snapshot(result, "after_load", model)

            result.phase = "generate"
            with gen_context_factory():
                result.generated_text = generate_fn(model)

            result.phase = "synchronize_after_generate"
            synchronize_fn()

            # After the operation: this is the snapshot that carries the rung's
            # true peak VRAM (the after_load one cannot, generation had not run)
            # and any ECC/Xid counters the operation itself moved.
            snapshot(result, "after_generate", model)

            result.status = "passed"
            result.phase = ""
            emit(f"  PASS in {clock() - started:.1f}s — generated: {result.generated_text!r}")
        except BaseException as exc:  # noqa: BLE001 - CUDA faults are not all Exception
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
            result.traceback = traceback.format_exc()
            aborted = True
            emit("")
            emit(f"  FAIL during phase '{result.phase}' after {clock() - started:.1f}s")
            emit(f"  {result.error}")

            # Post-fault evidence, best effort. torch's device queries will
            # likely fail on a poisoned CUDA context and come back `unavailable`
            # — but nvidia-smi and the kernel log are separate processes, and
            # the Xid code the driver just logged is the single most diagnostic
            # thing available. Not attempting to read it because torch is broken
            # would discard the evidence for being adjacent to the failure.
            try:
                synchronize_fn()
            except BaseException:  # noqa: BLE001 - already failing; this is expected to fail too
                pass
            snapshot(result, "on_failure", model)

            emit("")
            emit(f"  ISOLATION VERDICT — branch {step.index} ({step.name}):")
            emit(f"    {step.verdict_on_failure}")
        finally:
            result.seconds = clock() - started
            had_model = model is not None
            # Drop every reference this frame holds BEFORE teardown runs.
            # gen_context_factory is `model.disable_adapter` on rung 3 — a bound
            # method, which pins the model just as firmly as the model variable.
            model = None
            gen_context_factory = None
            if had_model:
                try:
                    release_fn()
                except Exception as exc:  # pragma: no cover - best effort teardown
                    emit(f"  (teardown after step {step.index} raised {exc!r}; continuing)")
            # Persist the summary after every rung, not once at the end: a rung
            # that kills the process must still leave the rungs before it on disk.
            try:
                progress_fn(results)
            except Exception as exc:  # pragma: no cover
                emit(f"  (progress write after step {step.index} raised {exc!r}; continuing)")

    return results


def matches_s0_fault_signature(error: str | None) -> bool:
    """Does this failure look like the fault the §0 probe actually died of?

    The probe died of `CUDA error: an illegal memory access was encountered`.
    A gated-repo 401, a disk-full OSError or a missing adapter revision are all
    failures at the same *configuration boundary* without being the same
    *fault*, and calling either one a reproduction would send someone hunting a
    CUDA bug that isn't there.
    """
    if not error:
        return False
    lowered = error.lower()
    return "illegal memory access" in lowered or (
        "cuda error" in lowered and "device-side assert" in lowered
    )


def summarise(results: list[StepResult]) -> dict:
    failed = next((r for r in results if r.status == "failed"), None)
    all_passed = bool(results) and all(r.status == "passed" for r in results)

    if failed is not None:
        outcome = "failed"
        verdict = failed.step.verdict_on_failure
        branch = failed.step.index
    elif all_passed:
        outcome = "all_passed"
        verdict = ALL_PASS_VERDICT
        branch = None
    else:
        # The case that matters most. This summary is rewritten after every
        # rung so it survives a hard death, which means "no rung has failed
        # yet" is a state it is routinely written in — and the previous version
        # rendered that state as ALL_PASS_VERDICT, i.e. "all four configurations
        # generated cleanly". A process killed during rung 2 left a durable
        # artifact claiming a clean sweep of rungs that never ran. A gate whose
        # crash residue reads as a pass is worse than no gate.
        outcome = "incomplete"
        verdict = INCOMPLETE_VERDICT
        branch = None

    # Deliberately two separate facts, because they answer different questions
    # and the old single `reproduces_s0_probe_crash` conflated them.
    within_configuration = failed is not None and failed.step.index <= 3
    signature_matches = matches_s0_fault_signature(failed.error) if failed else False

    if failed is None:
        reproduction = "not_applicable_no_failure"
    elif within_configuration and signature_matches:
        reproduction = "yes"
    elif signature_matches:
        reproduction = "same_fault_outside_the_probe_configuration"
    elif within_configuration:
        reproduction = "no_different_fault"
    else:
        reproduction = "no"

    return {
        "schema": "isolation_ladder/v2",
        "outcome": outcome,
        "category": PROBE_CATEGORY,
        "expected_prompt_tokens": EXPECTED_PROMPT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        "base_model": {"repo": BASE_MODEL_REPO, "revision": BASE_MODEL_REVISION},
        "adapter": {
            "repo": SFT_ADAPTER_REPO,
            "subfolder": SFT_ADAPTER_SUBFOLDER,
            "revision": SFT_ADAPTER_REVISION,
        },
        "failed_at_step": branch,
        "failed_phase": failed.phase if failed else None,
        # A fact about which configurations were exercised — NOT a claim that
        # the §0 crash was reproduced.
        "failed_within_probe_configuration": within_configuration,
        "fault_signature_matches_s0": signature_matches,
        "s0_reproduction": reproduction,
        "verdict": verdict,
        "steps": [r.to_dict() for r in results],
    }


# ---------------------------------------------------------------------------
# Real wiring
# ---------------------------------------------------------------------------
def _real_synchronize():
    import torch

    torch.cuda.synchronize()


def _real_reset_peak():
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _real_release():
    """Takes no model, by design — see run_ladder's docstring. The caller has
    already dropped every reference it held; this only has to collect and hand
    the freed blocks back to the driver."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _real_generate(model, tokenizer, prompt: str) -> str:
    """Production `generate` from bfcl_simple, at 8 tokens."""
    from eval.bfcl_simple import generate

    return generate(model, tokenizer, prompt, MAX_NEW_TOKENS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="§0 smoke gate: four-rung isolation ladder for the CUDA illegal access."
    )
    # Required for a real run, not optional. Evidence that can be lost is not
    # evidence: the whole reason this exists is that the last run died and left
    # nothing behind.
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for the summary, per-rung telemetry, and the raw nvidia-smi dump. "
        "Required unless --dry-run. Put it on the persistent volume.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ladder and exit. Loads nothing, needs no GPU, costs nothing.",
    )
    # There is deliberately no --expect-prompt-tokens flag. An operator able to
    # bless an arbitrary prompt length can turn the gate green on a prompt that
    # is not the one that crashed, which is the same as having no gate while
    # the docs promise one. Tests inject the value through the function
    # parameter instead.
    return parser


def print_plan(emit: Callable[[str], None] = print) -> None:
    emit("Isolation ladder — one variable per rung, abort on first failure:")
    for step in LADDER:
        emit(f"  Step {step.index}: {step.name}")
    emit("")
    emit("  1 vs 2 isolates placement | 2 vs 3 isolates the PEFT wrapper | 3 vs 4 isolates adapter state")
    emit(f"  prompt: first item of category={PROBE_CATEGORY}, asserted at {EXPECTED_PROMPT_TOKENS} tokens")
    emit(f"  generation: {MAX_NEW_TOKENS} new tokens, greedy, per step")
    emit("  CUDA_LAUNCH_BLOCKING=1, torch.cuda.synchronize() after load and after generate")
    emit(f"  base: {BASE_MODEL_REPO}@{BASE_MODEL_REVISION}")
    emit(f"  adapter: {SFT_ADAPTER_REPO}/{SFT_ADAPTER_SUBFOLDER}@{SFT_ADAPTER_REVISION}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)

    if args.dry_run:
        print_plan()
        print("\nDRY RUN: nothing loaded, no GPU touched, no spend.")
        return EXIT_OK

    if args.out_dir is None:
        print(
            "ERROR: --out-dir is required for a real run. The rung that kills the\n"
            "       process is the one whose evidence matters, and evidence held in\n"
            "       memory until the ladder returns is exactly the evidence that run\n"
            "       never produces.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    if needs_launch_blocking_reexec(os.environ):
        reexec_with_launch_blocking([__file__, *argv], os.environ)
        return EXIT_OK  # unreachable after a successful execve

    print_plan()

    out_dir: Path = args.out_dir
    telemetry_dir = out_dir / "telemetry"
    summary_path = out_dir / "isolation_ladder.json"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    # Written once, before anything is loaded: the pre-run state of the card is
    # the baseline every later reading is compared against, and it is also the
    # only reading guaranteed to exist if load itself kills the process.
    smi_status = gpu_telemetry.write_raw_smi_query(out_dir / "nvidia_smi_q_pre_run.txt")
    print(f"\nraw nvidia-smi -q: {smi_status['status']}")
    baseline = gpu_telemetry.collect_all(model=None, include_smi_raw=False, phase="pre_run")
    baseline["libraries"] = gpu_telemetry.collect_library_versions()
    gpu_telemetry.write_json_atomic(telemetry_dir / "step0_pre_run.json", baseline)

    print("\nLoading tokenizer and building the first production prompt...")
    tokenizer = _load_tokenizer()
    prompt_id, prompt = load_first_production_prompt(tokenizer)
    try:
        token_count = assert_prompt_token_count(tokenizer, prompt)
    except ValueError as exc:
        print(f"\nREFUSING TO RUN: {exc}", file=sys.stderr)
        return EXIT_PROMPT_MISMATCH
    print(f"  prompt id={prompt_id}, {token_count} tokens (asserted)")

    def persist_snapshot(step: LadderStep, phase: str, model: Any) -> dict:
        bundle = gpu_telemetry.collect_all(model, phase=phase)
        gpu_telemetry.write_json_atomic(
            telemetry_dir / f"step{step.index}_{phase}.json", bundle
        )
        return bundle

    def persist_summary(results: list[StepResult]) -> None:
        partial = summarise(results)
        partial["prompt_id"] = prompt_id
        partial["prompt_tokens"] = token_count
        # Derived from `outcome`, never recomputed independently — two
        # completeness notions that can disagree is how the false all-pass got
        # written in the first place.
        partial["complete"] = partial["outcome"] != "incomplete"
        gpu_telemetry.write_json_atomic(summary_path, partial)

    # An incomplete summary on disk before rung 1 even starts. If the process
    # dies during the first load, the artifact says INCOMPLETE — NO CONCLUSION
    # rather than not existing at all, and an absent file is the one thing a
    # reader is most likely to interpret as "the gate was not reached" when in
    # fact it was and it died.
    persist_summary([StepResult(step=s) for s in LADDER])

    results = run_ladder(
        generate_fn=lambda model: _real_generate(model, tokenizer, prompt),
        snapshot_fn=persist_snapshot,
        synchronize_fn=_real_synchronize,
        release_fn=_real_release,
        reset_peak_fn=_real_reset_peak,
        progress_fn=persist_summary,
    )

    summary = summarise(results)
    summary["prompt_id"] = prompt_id
    summary["prompt_tokens"] = token_count
    summary["complete"] = summary["outcome"] != "incomplete"
    summary["raw_nvidia_smi_q"] = smi_status
    gpu_telemetry.write_json_atomic(summary_path, summary)

    print("")
    print("=" * 72)
    print("LADDER RESULT")
    print("=" * 72)
    for result in results:
        print(f"  step {result.step.index}: {result.status:<8} {result.step.name}")
    print("")
    print(summary["verdict"])
    print(f"\nwrote {summary_path}")
    print(f"      {telemetry_dir}/ (per-rung snapshots)")

    # Green ONLY on an explicit all-pass. `failed_at_step is None` was the old
    # condition and it is true for an incomplete ladder too, which would have
    # let the launcher's gate open on a run that never finished.
    return EXIT_OK if summary["outcome"] == "all_passed" else EXIT_LADDER_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
