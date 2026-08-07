"""Pinned, deterministic generation backend for the study-2 miner.

This module owns model-facing correctness: exact weights, exact prompt turns,
one genuinely batched generation call per prompt, and an exclusive run lock.
It deliberately contains no prices, budgets, estimates, or provider billing
logic. Those belong to the launch procedure; the miner's artifacts should say
what was sampled, not pretend to reproduce a provider invoice.

Torch, PEFT, and Transformers remain lazy imports so every non-model decision is
testable with no GPU, weights, network, or paid infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mining.mine_pairs import (
    BASE_MODEL_REPO,
    BASE_MODEL_REVISION,
    MAX_NEW_TOKENS,
    SAMPLING_MODE,
    SEED,
    SEED_DERIVATION,
    SFT_ADAPTER_REPO,
    SFT_ADAPTER_REVISION,
    SFT_ADAPTER_SUBFOLDER,
    TEMPERATURE,
    TOP_P,
    MinerError,
    Prompt,
)

LOCK_NAME = "run.lock"


class BackendError(MinerError):
    """Generation cannot proceed reproducibly, so it does not proceed."""


def derive_batch_seed(prompt_id: str, seed: int = SEED) -> int:
    """Derive one resume-stable seed for a prompt's complete sample batch."""
    digest = hashlib.sha256(f"{seed}:{prompt_id}:batch".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def build_chat_prompt(tokenizer: Any, prompt: Prompt) -> str:
    """Render the row's exact pre-target turns through the pinned template."""
    messages = prompt.prompt_messages
    if not messages:
        raise BackendError(f"{prompt.prompt_id}: no prompt turns to render")
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def preflight_chat_template(tokenizer: Any, prompts: Sequence[Prompt]) -> int:
    """Render every selected prompt before weights load, failing closed."""
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


class RunLock:
    """Allow only one model process to extend a stage directory."""

    def __init__(self, out_dir: Path) -> None:
        self.path = Path(out_dir) / LOCK_NAME

    def __enter__(self) -> RunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError as exc:
            holder = self.path.read_text().strip() or "unknown"
            raise BackendError(
                f"{self.path} is held by {holder}. Another process may be sampling "
                "this stage. If it is certainly dead, remove the stale lock "
                "deliberately; it is never stolen automatically"
            ) from exc
        with os.fdopen(fd, "w") as handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.path.unlink(missing_ok=True)


def make_generate_fn(
    model: Any,
    tokenizer: Any,
    generate_batch: Callable[
        [Any, Any, str, int, float, float, int, int], list[str]
    ],
) -> Callable[[Prompt, int], list[str]]:
    """Build the miner callback around a mandatory batch implementation.

    There is intentionally no sequential fallback. A caller that forgets to
    wire the production batch cannot silently make eight separate model calls.
    """

    def generate_fn(prompt: Prompt, samples: int) -> list[str]:
        if samples <= 0:
            raise BackendError(f"{prompt.prompt_id}: sample count must be positive")
        rendered = build_chat_prompt(tokenizer, prompt)
        outputs = generate_batch(
            model,
            tokenizer,
            rendered,
            MAX_NEW_TOKENS,
            TEMPERATURE,
            TOP_P,
            derive_batch_seed(prompt.prompt_id),
            samples,
        )
        if len(outputs) != samples:
            raise BackendError(
                f"{prompt.prompt_id}: batch returned {len(outputs)} of "
                f"{samples} requested samples"
            )
        if not all(isinstance(output, str) for output in outputs):
            raise BackendError(f"{prompt.prompt_id}: batch returned a non-text sample")
        return outputs

    return generate_fn


def load_tokenizer() -> Any:
    """Load the exact tokenizer pinned by the preregistration."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_REPO, revision=BASE_MODEL_REVISION
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_policy_with(
    tokenizer: Any, device_map: str = "auto"
) -> Callable[[Prompt, int], list[str]]:
    """Load the pinned policy and wire its production batched generation path."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    preflight_adapter()

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_REPO,
        revision=BASE_MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(
        base,
        SFT_ADAPTER_REPO,
        subfolder=SFT_ADAPTER_SUBFOLDER.rstrip("/"),
        revision=SFT_ADAPTER_REVISION,
    ).eval()

    def generate_batch(
        model: Any,
        tokenizer: Any,
        rendered: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        samples: int,
    ) -> list[str]:
        torch.manual_seed(seed)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            sequences = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=samples,
                pad_token_id=tokenizer.pad_token_id,
            )
        return [
            tokenizer.decode(sequence[prompt_tokens:], skip_special_tokens=True)
            for sequence in sequences
        ]

    return make_generate_fn(model, tokenizer, generate_batch)


def preflight_adapter() -> Any:
    """Resolve adapter metadata before any base-model weight is loaded."""
    from peft import PeftConfig

    try:
        return PeftConfig.from_pretrained(
            SFT_ADAPTER_REPO,
            subfolder=SFT_ADAPTER_SUBFOLDER.rstrip("/"),
            revision=SFT_ADAPTER_REVISION,
        )
    except Exception as exc:
        raise BackendError(
            f"adapter preflight failed for {SFT_ADAPTER_REPO}/{SFT_ADAPTER_SUBFOLDER}: "
            f"{exc}"
        ) from exc


def verify_persistent_root(
    out_dir: Path, persistent_root: Path | None, attested: bool = False
) -> Path:
    """Require paid model artifacts to live under a durable mounted root."""
    if persistent_root is None:
        raise BackendError(
            "--persistent-root is required for a model run: container disk is "
            "erased when the pod is terminated"
        )
    root = Path(persistent_root).resolve()
    if not root.is_dir():
        raise BackendError(f"--persistent-root {root} is not an existing directory")
    if not attested and os.stat(root).st_dev == os.stat("/").st_dev:
        raise BackendError(
            f"--persistent-root {root} is on the container root device. Point it "
            "at a mounted volume, or pass --attest-durable-root only after "
            "independently confirming that this path survives pod termination"
        )

    probe = root / f".mine_pairs_write_probe.{os.getpid()}"
    try:
        probe.write_text("probe\n")
        probe.unlink()
    except OSError as exc:
        raise BackendError(f"--persistent-root {root} is not writable: {exc}") from exc

    resolved = Path(out_dir).resolve()
    if root not in resolved.parents and resolved != root:
        raise BackendError(
            f"--out-dir {resolved} is not inside --persistent-root {root}"
        )
    return resolved


def execute_model_stage(
    out_dir: Path,
    stage: str,
    persistent_root: Path | None,
    fresh: bool = False,
    attested: bool = False,
) -> dict[str, Any]:
    """Run the one model-backed path after all non-model checks pass.

    Provider auto-stop and a hard runaway timeout are deliberately external.
    This function does not accept or calculate dollars, rates, or estimates.
    """
    from mining.mine_pairs import _run_prepared, prepare_stage_run

    context_path = os.environ.get("LLAMA_TOOLS_LAUNCH_CONTEXT")
    context_token = os.environ.get("LLAMA_TOOLS_LAUNCH_TOKEN")
    if not context_path or not context_token:
        raise BackendError(
            "refusing direct model launch: invoke scripts/launch_mining_stage.py "
            "so the external lifecycle envelope is present"
        )
    try:
        context = json.loads(Path(context_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError(f"invalid launcher context {context_path}: {exc}") from exc
    if context.get("token") != context_token:
        raise BackendError("launcher context token does not match")
    if Path(context.get("out_dir", "")).resolve() != Path(out_dir).resolve():
        raise BackendError("launcher context does not match --out-dir")

    resolved_out = verify_persistent_root(out_dir, persistent_root, attested)
    with RunLock(resolved_out):
        # This binds run.json, including the backend digest and batch semantics,
        # before tokenizer or model load. A mismatched resume fails at $0 here.
        preflight = prepare_stage_run(resolved_out, stage, fresh)
        tokenizer = load_tokenizer()
        rendered = preflight_chat_template(tokenizer, preflight.selected)
        print(f"chat-template preflight rendered {rendered} prompts before model load")
        generate_fn = load_policy_with(tokenizer)
        return _run_prepared(resolved_out, preflight, generate_fn)


def backend_sha256() -> str:
    """Digest recorded in immutable run identity."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


__all__ = [
    "SAMPLING_MODE",
    "SEED_DERIVATION",
    "BackendError",
    "RunLock",
    "backend_sha256",
    "build_chat_prompt",
    "derive_batch_seed",
    "execute_model_stage",
    "load_policy_with",
    "load_tokenizer",
    "make_generate_fn",
    "preflight_adapter",
    "preflight_chat_template",
    "verify_persistent_root",
]
