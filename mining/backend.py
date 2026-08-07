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
from collections.abc import Callable
from dataclasses import dataclass
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
    if hourly_rate_usd <= 0:
        raise BackendError("hourly rate must be positive to bound a run")
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

    deadline_seconds: int
    monotonic: Callable[[], float]
    started_at: float = 0.0
    prompts_started: int = 0

    def __post_init__(self) -> None:
        if self.deadline_seconds <= 0:
            raise BackendError("a run with no positive deadline may not start")
        self.started_at = self.monotonic()

    @property
    def elapsed(self) -> float:
        return self.monotonic() - self.started_at

    @property
    def remaining(self) -> float:
        return self.deadline_seconds - self.elapsed

    def check(self) -> None:
        """Refuse to start further work once the budget is spent."""
        if self.remaining <= 0:
            raise BackendError(
                f"spend deadline reached after {self.elapsed:.0f}s and "
                f"{self.prompts_started} prompts; stopping before the next prompt. "
                f"The ledger holds every prompt already completed — rerun the same "
                f"command against the same directory to resume"
            )
        self.prompts_started += 1


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
        guard.check()
        rendered = build_chat_prompt(tokenizer, prompt)
        outputs: list[str] = []
        for index in range(samples):
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


def load_policy(hourly_rate_usd: float, cap_usd: float, device_map: str = "auto") -> Any:
    """Load the pinned SFT policy and return a bounded `generate_fn`.

    The first line in this project that can spend money, which is why it is the
    last thing built and why it takes the rate as an argument: the deadline is
    computed from what this pod actually charges, not from a constant.
    """
    # Deliberately lazy: importing torch is not needed to reason about a run.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    deadline = compute_deadline_seconds(cap_usd, hourly_rate_usd)
    import time

    guard = SpendGuard(deadline_seconds=deadline, monotonic=time.monotonic)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_REPO, revision=BASE_MODEL_REVISION
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    return make_generate_fn(model, tokenizer, guard, generate_once), guard


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
