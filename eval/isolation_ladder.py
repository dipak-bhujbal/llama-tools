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
    status: str = "pending"  # pending | passed | failed | skipped
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
    telemetry_fn: Callable[[Any], dict],
    synchronize_fn: Callable[[], None],
    release_fn: Callable[[Any], None],
    reset_peak_fn: Callable[[], None] = lambda: None,
    emit: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
) -> list[StepResult]:
    """Run the rungs in order, aborting at the first failure.

    Returns one StepResult per step; steps after a failure are 'skipped', which
    is recorded explicitly rather than omitted — a reader must be able to tell
    "step 4 was not run" from "step 4 has no result".
    """
    loaders = LOADERS if loaders is None else loaders
    results = [StepResult(step=s) for s in steps]
    aborted = False

    for result in results:
        step = result.step
        if aborted:
            result.status = "skipped"
            result.error = "not run: an earlier rung failed and the CUDA context is unusable"
            continue

        emit("")
        emit("=" * 72)
        emit(f"STEP {step.index}/{len(steps)}: {step.name}")
        emit("=" * 72)

        reset_peak_fn()
        started = clock()
        model = None
        try:
            result.phase = "load"
            model, gen_context_factory = loaders[step.loader]()

            result.phase = "synchronize_after_load"
            synchronize_fn()

            # Telemetry is captured after load and before generation: this is the
            # only moment where the model exists, its placement is resolved, and
            # nothing has faulted yet. Capturing it later would mean capturing
            # nothing on the step that crashes — which is what happened to the
            # probe.
            result.telemetry = telemetry_fn(model)
            gaps = gpu_telemetry.unavailable_fields(result.telemetry)
            if gaps:
                emit(f"  telemetry gaps ({len(gaps)}) — these are unmeasured, not zero:")
                for gap in gaps:
                    emit(f"    - {gap}")

            result.phase = "generate"
            with gen_context_factory():
                result.generated_text = generate_fn(model)

            result.phase = "synchronize_after_generate"
            synchronize_fn()

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
            emit("")
            emit(f"  ISOLATION VERDICT — branch {step.index} ({step.name}):")
            emit(f"    {step.verdict_on_failure}")
        finally:
            result.seconds = clock() - started
            if model is not None:
                try:
                    release_fn(model)
                except Exception as exc:  # pragma: no cover - best effort teardown
                    emit(f"  (teardown after step {step.index} raised {exc!r}; continuing)")

    return results


def summarise(results: list[StepResult]) -> dict:
    failed = next((r for r in results if r.status == "failed"), None)
    if failed is None:
        verdict = ALL_PASS_VERDICT
        reproduced = False
        branch = None
    else:
        verdict = failed.step.verdict_on_failure
        # Step 3 is the probe's own configuration; failing there (or earlier) is
        # a reproduction of the §0 crash. Failing only at step 4 is not — the
        # probe never reached an adapter-enabled candidate.
        reproduced = failed.step.index <= 3
        branch = failed.step.index
    return {
        "schema": "isolation_ladder/v1",
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
        "reproduces_s0_probe_crash": reproduced,
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


def _real_release(model):
    import torch

    del model
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
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the JSON result here (default: stdout only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ladder and exit. Loads nothing, needs no GPU, costs nothing.",
    )
    parser.add_argument(
        "--expect-prompt-tokens",
        type=int,
        default=EXPECTED_PROMPT_TOKENS,
        help=f"Required token count for the first prompt (default {EXPECTED_PROMPT_TOKENS}).",
    )
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

    if needs_launch_blocking_reexec(os.environ):
        reexec_with_launch_blocking([__file__, *argv], os.environ)
        return EXIT_OK  # unreachable after a successful execve

    print_plan()

    print("\nLoading tokenizer and building the first production prompt...")
    tokenizer = _load_tokenizer()
    prompt_id, prompt = load_first_production_prompt(tokenizer)
    try:
        token_count = assert_prompt_token_count(tokenizer, prompt, args.expect_prompt_tokens)
    except ValueError as exc:
        print(f"\nREFUSING TO RUN: {exc}", file=sys.stderr)
        return EXIT_PROMPT_MISMATCH
    print(f"  prompt id={prompt_id}, {token_count} tokens (asserted)")

    results = run_ladder(
        generate_fn=lambda model: _real_generate(model, tokenizer, prompt),
        telemetry_fn=lambda model: gpu_telemetry.collect_all(model),
        synchronize_fn=_real_synchronize,
        release_fn=_real_release,
        reset_peak_fn=_real_reset_peak,
    )

    summary = summarise(results)
    summary["prompt_id"] = prompt_id
    summary["prompt_tokens"] = token_count

    print("")
    print("=" * 72)
    print("LADDER RESULT")
    print("=" * 72)
    for result in results:
        print(f"  step {result.step.index}: {result.status:<8} {result.step.name}")
    print("")
    print(summary["verdict"])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.out}")

    return EXIT_OK if summary["failed_at_step"] is None else EXIT_LADDER_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
