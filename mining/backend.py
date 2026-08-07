"""Generation backend for the miner — the first path that can cost money.

Everything here exists to make sampling *reproducible* and *mechanically
bounded*. The miner itself (`mining/mine_pairs.py`) never imports a model: it
takes a `generate_fn`, and this module is the only thing that builds a real one.

Three properties it has to guarantee, none of which the miner can enforce alone:

**Resume-stable seeding.** A run that crashes after 40 prompts and resumes must
produce, for prompt 41, exactly what it would have produced without the crash.
A single global RNG advanced once per generation cannot do that — after a resume
it has been advanced a different number of times, so the "seeded" run silently
samples something else. The seed is therefore derived per `(prompt_id, sample
index)` from the frozen §2.4 seed, and never from call order.

**A deadline the run cannot outlive.** The owner approved a total-stage cap, and
a cap that depends on someone watching a console is not a cap. `SpendGuard`
holds a wall-clock deadline derived from the cap and the *actual* hourly rate,
and refuses to start another prompt past it. Construction without a deadline
raises — there is no unbounded mode to fall into.

**Pinned weights.** Base model and adapter come from §2.1's pinned repo,
subfolder and revision. Nothing here accepts an arbitrary checkpoint path.

Torch and transformers are imported lazily, inside the factory, so this module
and its tests stay importable — and runnable — with no GPU and no weights.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mining.mine_pairs import (
    BASE_MODEL_REPO,
    BASE_MODEL_REVISION,
    MAX_NEW_TOKENS,
    SAMPLES_PER_PROMPT,
    SEED,
    SFT_ADAPTER_REPO,
    SFT_ADAPTER_REVISION,
    SFT_ADAPTER_SUBFOLDER,
    TEMPERATURE,
    TOP_P,
    MinerError,
    Prompt,
)

# Reserved out of the owner's total-stage cap for storage and shutdown, so the
# compute deadline cannot consume the entire budget and leave the tail unpaid.
STORAGE_RESERVE_USD = 0.08

# No stage may hold a pod longer than this regardless of how cheap the hour is:
# the estimate's worst case is 1.5 h, and authorizing more wall-clock than the
# estimate covers would make the estimate decorative.
MAX_WALL_CLOCK_SECONDS = 2 * 60 * 60

# The owner approved exactly this much (msgs 2370-2371). A cap argument is an
# operator transcription of that decision, not a place to raise it.
APPROVED_CAP_USD = 1.00


class BackendError(MinerError):
    """Generation cannot proceed safely, so it does not proceed."""


def derive_prompt_seed(prompt_id: str, sample_index: int, seed: int = SEED) -> int:
    """A seed determined by identity, never by call order.

    Resume is the whole reason this is not `seed + counter`. After a crash the
    global RNG has been advanced a different number of times, so an order-derived
    seed reproduces nothing; an identity-derived one reproduces everything.
    """
    digest = hashlib.sha256(f"{seed}:{prompt_id}:{sample_index}".encode()).digest()
    # 63 bits: torch.manual_seed rejects values at or above 2**64, and staying
    # inside the signed range keeps the value portable across backends.
    return int.from_bytes(digest[:8], "big") >> 1


def compute_deadline_seconds(cap_usd: float, hourly_rate_usd: float) -> int:
    """Seconds of compute the cap buys at the rate actually charged.

    Takes the *observed* console rate rather than a remembered one. July's
    $0.49/hr is evidence of what was paid then, not of what this pod costs.
    """
    for label, value in (("cap", cap_usd), ("hourly rate", hourly_rate_usd)):
        if value != value or value in (float("inf"), float("-inf")):
            raise BackendError(f"{label} must be a finite number, not {value!r}")
    if hourly_rate_usd <= 0:
        raise BackendError("hourly rate must be positive to bound a run")
    if cap_usd > APPROVED_CAP_USD:
        raise BackendError(
            f"cap ${cap_usd:.2f} exceeds the owner-approved ${APPROVED_CAP_USD:.2f} "
            f"ceiling; raising it is an owner decision, not a flag"
        )
    budget = cap_usd - STORAGE_RESERVE_USD
    if budget <= 0:
        raise BackendError(
            f"cap ${cap_usd:.2f} does not cover the ${STORAGE_RESERVE_USD:.2f} "
            f"storage reserve; this configuration cannot launch"
        )
    return min(MAX_WALL_CLOCK_SECONDS, int(budget / hourly_rate_usd * 3600))


@dataclass
class SpendGuard:
    """A wall-clock bound the run cannot outlive.

    `monotonic` is injected so tests can drive it without sleeping, and so the
    guard cannot be defeated by a wall-clock adjustment mid-run.
    """

    deadline_seconds: float
    monotonic: Callable[[], float]
    consumed_before: float = 0.0
    started_at: float = 0.0
    prompts_started: int = 0
    samples_started: int = 0

    def __post_init__(self) -> None:
        if self.deadline_seconds <= 0:
            raise BackendError("a run with no positive deadline may not start")
        self.started_at = self.monotonic()

    @property
    def elapsed(self) -> float:
        """This session's elapsed time plus everything earlier sessions billed."""
        return (self.monotonic() - self.started_at) + self.consumed_before

    @property
    def remaining(self) -> float:
        return self.deadline_seconds - self.elapsed

    def check(self, unit: str = "prompt") -> None:
        """Refuse to start further work once the budget is spent.

        Called before every **sample**, not only every prompt. Checking once per
        prompt bounds nothing when a prompt is 8 sequential generations: codex
        drove 8 samples of 100s each past a 10s deadline and all 8 ran.
        """
        if self.remaining <= 0:
            raise BackendError(
                f"spend deadline reached after {self.elapsed:.0f}s of the "
                f"{self.deadline_seconds:.0f}s stage allowance "
                f"({self.prompts_started} prompts, {self.samples_started} samples); "
                f"stopping before the next {unit}. Every completed prompt is in the "
                f"ledger — rerun the same command against the same directory to "
                f"resume within whatever allowance remains"
            )
        if unit == "prompt":
            self.prompts_started += 1
        else:
            self.samples_started += 1


def build_chat_prompt(tokenizer: Any, prompt: Prompt) -> str:
    """Render the row's own preceding turns through the model's chat template.

    Uses `prompt.prompt_messages` — the exact turns before the first assistant
    turn, roles intact — rather than a rebuilt system+user pair. The 7 pool rows
    carrying a `tool` message before their first assistant depend on that
    difference; flattening drops the context their completion answers.
    """
    messages = prompt.prompt_messages
    if not messages:
        raise BackendError(f"{prompt.prompt_id}: no prompt turns to render")
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def make_generate_fn(
    model: Any,
    tokenizer: Any,
    guard: SpendGuard,
    generate_once: Callable[[Any, Any, str, int, float, float, int], str],
) -> Callable[[Prompt, int], list[str]]:
    """Assemble the miner's `generate_fn` from an already-loaded model.

    Split from `load_policy()` so every rule above — seeding, the deadline, the
    chat rendering — is testable without weights. `generate_once` is the only
    part that touches a GPU.
    """

    def generate_fn(prompt: Prompt, samples: int) -> list[str]:
        guard.check("prompt")
        rendered = build_chat_prompt(tokenizer, prompt)
        outputs: list[str] = []
        for index in range(samples):
            # Per sample, not per prompt: a prompt is 8 generations, and a bound
            # that is only consulted between prompts does not bound them.
            guard.check("sample")
            outputs.append(
                generate_once(
                    model,
                    tokenizer,
                    rendered,
                    MAX_NEW_TOKENS,
                    TEMPERATURE,
                    TOP_P,
                    derive_prompt_seed(prompt.prompt_id, index),
                )
            )
        return outputs

    return generate_fn


def preflight_chat_template(tokenizer: Any, prompts: Sequence[Prompt]) -> int:
    """Render every selected prompt before a model is loaded, or refuse.

    The pinned Llama-3.1 template may reject a bare `tool` role, and 7 of the
    pool's rows carry one. Discovering that after the weights are resident means
    discovering it on billed time; the tokenizer alone is free to load.
    """
    for prompt in prompts:
        try:
            rendered = build_chat_prompt(tokenizer, prompt)
        except BackendError:
            raise
        except Exception as exc:
            roles = [role for role, _ in prompt.messages]
            raise BackendError(
                f"{prompt.prompt_id}: the pinned chat template cannot render this "
                f"row's turns {roles}: {exc}. Refusing before any weights load"
            ) from exc
        if not rendered.strip():
            raise BackendError(f"{prompt.prompt_id}: chat template rendered empty")
    return len(prompts)


LOCK_NAME = "run.lock"


class RunLock:
    """One process at a time may bill against a stage directory.

    Two concurrent miners would duplicate paid generations and consume the
    ceiling twice while each believed itself compliant. `O_EXCL` makes the claim
    atomic; a stale lock is reported rather than stolen, because "the other
    process is probably dead" is exactly the assumption that produces two live
    miners.
    """

    def __init__(self, out_dir: Path) -> None:
        self.path = Path(out_dir) / LOCK_NAME
        self._fd: int | None = None

    def __enter__(self) -> RunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError as exc:
            holder = self.path.read_text().strip() or "unknown"
            raise BackendError(
                f"{self.path} is held by {holder}. Another process may be billing "
                f"against this stage. If it is certainly dead, remove the lock "
                f"deliberately — this is not stolen automatically"
            ) from exc
        with os.fdopen(self._fd, "w") as handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._fd = None
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.path.unlink(missing_ok=True)


def load_tokenizer() -> Any:
    """Load only the tokenizer. Free, and enough to prove the template renders."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_REPO, revision=BASE_MODEL_REVISION
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_policy_with(
    tokenizer: Any, guard: SpendGuard, device_map: str = "auto"
) -> Callable[[Prompt, int], list[str]]:
    """Load the pinned SFT policy and return a `generate_fn` bound by `guard`.

    The first line in this project that can spend money, which is why it is the
    last thing reached: the allowance, the lock, and the chat template are all
    proved before these weights are touched.
    """
    # Deliberately lazy: importing torch is not needed to reason about a run.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_REPO,
        revision=BASE_MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(
        base,
        SFT_ADAPTER_REPO,
        subfolder=SFT_ADAPTER_SUBFOLDER,
        revision=SFT_ADAPTER_REVISION,
    ).eval()

    def generate_once(
        model: Any,
        tokenizer: Any,
        rendered: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> str:
        torch.manual_seed(seed)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    return make_generate_fn(model, tokenizer, guard, generate_once)


def sampling_receipt(hourly_rate_usd: float, cap_usd: float) -> dict[str, Any]:
    """What a launch sheet must record before the pod starts billing."""
    return {
        "base_model": {"repo": BASE_MODEL_REPO, "revision": BASE_MODEL_REVISION},
        "adapter": {
            "repo": SFT_ADAPTER_REPO,
            "subfolder": SFT_ADAPTER_SUBFOLDER,
            "revision": SFT_ADAPTER_REVISION,
        },
        "sampling": {
            "samples": SAMPLES_PER_PROMPT,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "seed": SEED,
            "seed_derivation": "sha256(seed:prompt_id:sample_index) >> 1",
        },
        "spend_bound": {
            "cap_usd": cap_usd,
            "hourly_rate_usd": hourly_rate_usd,
            "storage_reserve_usd": STORAGE_RESERVE_USD,
            "deadline_seconds": compute_deadline_seconds(cap_usd, hourly_rate_usd),
            "max_wall_clock_seconds": MAX_WALL_CLOCK_SECONDS,
        },
    }


def verify_persistent_root(out_dir: Path, persistent_root: Path | None) -> Path:
    """Refuse a container-local output directory for a billed run.

    A pilot whose only copy of the ledger lives on the pod's container disk is
    one `stop` away from having cost money and produced nothing — container disk
    is erased on stop, which is exactly why it is the cheap option. The operator
    must name a durable mount and the output must sit inside it.
    """
    if persistent_root is None:
        raise BackendError(
            "--persistent-root is required for a paid run: the ledger is the only "
            "evidence the money bought, and container disk is erased on stop"
        )
    root = Path(persistent_root).resolve()
    if not root.is_dir():
        raise BackendError(f"--persistent-root {root} is not an existing directory")
    probe = root / ".mine_pairs_write_probe"
    try:
        probe.write_text("probe\n")
        probe.unlink()
    except OSError as exc:
        raise BackendError(f"--persistent-root {root} is not writable: {exc}") from exc

    resolved = Path(out_dir).resolve()
    if root not in resolved.parents and resolved != root:
        raise BackendError(
            f"--out-dir {resolved} is not inside --persistent-root {root}; a billed "
            f"run may not write its only evidence to container-local storage"
        )
    return resolved


def execute_paid_stage(
    out_dir: Path,
    stage: str,
    hourly_rate_usd: float,
    cap_usd: float,
    persistent_root: Path | None,
    fresh: bool = False,
    now: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """The one billable entry point, with every bound applied in order.

    Order matters and is the point: the allowance is claimed and the template is
    proved *before* weights load, so the expensive failure modes are discovered
    while the pod is still cheap.
    """
    import time

    from mining.mine_pairs import RUN_METADATA_NAME, preflight_stage, run
    from mining.spend_ledger import close_session, open_session

    clock = now or time.time
    resolved_out = verify_persistent_root(out_dir, persistent_root)
    total_seconds = compute_deadline_seconds(cap_usd, hourly_rate_usd)

    with RunLock(resolved_out):
        session, allowance = open_session(
            resolved_out, total_seconds, hourly_rate_usd, cap_usd, clock()
        )
        print(
            f"stage allowance {total_seconds}s; already consumed "
            f"{allowance.consumed_seconds:.0f}s across {allowance.sessions} session(s); "
            f"{allowance.remaining_seconds:.0f}s remain at ${hourly_rate_usd:.2f}/hr"
        )

        exit_reason = "unknown"
        try:
            preflight = preflight_stage(stage)
            tokenizer = load_tokenizer()
            rendered = preflight_chat_template(tokenizer, preflight.selected)
            print(f"chat-template preflight rendered {rendered} prompts before model load")

            guard = SpendGuard(
                deadline_seconds=total_seconds,
                monotonic=time.monotonic,
                consumed_before=allowance.consumed_seconds,
            )
            generate_fn = load_policy_with(tokenizer, guard)
            summary = run(out_dir=resolved_out, stage=stage, generate_fn=generate_fn)
            exit_reason = "completed"
            return summary
        except BackendError as exc:
            exit_reason = f"stopped: {exc}"
            raise
        except BaseException as exc:
            exit_reason = f"failed: {type(exc).__name__}"
            raise
        finally:
            record = close_session(
                resolved_out, session, clock(), exit_reason, hourly_rate_usd
            )
            receipt = sampling_receipt(hourly_rate_usd, cap_usd)
            receipt["session"] = record
            receipt["run_metadata"] = str(resolved_out / RUN_METADATA_NAME)
            receipt["backend_sha256"] = _sha256_of_backend()
            from mining.mine_pairs import _atomic_write

            _atomic_write(
                resolved_out / "spend_receipt.json",
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            )
            print(
                f"session {session} ended ({exit_reason}); stage consumed "
                f"{record['stage_consumed_seconds']:.0f}s "
                f"≈ ${record['stage_estimated_cost_usd']:.4f}"
            )


def _sha256_of_backend() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
