"""Week 5 DPO smoke: verify the DPO loop wires up before the Week 6 full run.

Scale-down analog of `train/sft_smoke.py` for DPO. 500 preference pairs from
the Week 3 preference set, ~30 min on 1x RTX A6000 48GB. Loads the Week 4
SFT LoRA adapter (`outputs/sft-full`) on top of the base model as the policy.

Reference-model handling: we use TRL's PEFT-adapter DPO pattern where the
reference is derived implicitly by disabling the LoRA adapter on the policy
model (DPOTrainer, when passed a PEFT model with `ref_model=None`, uses the
base model — i.e., the adapter turned off — as the reference). This avoids
loading two full 8B copies onto a single 48GB card.

Prereqs on the pod:
- Week 4 SFT complete; `outputs/sft-full/` present with the LoRA adapter files.
- `pip install -e ".[train]"` completed
- `hf auth login --token $HF_TOKEN` succeeded
- `wandb login $WANDB_API_KEY` (optional)
- `data/processed/preferences_dpo.jsonl` present on the pod

Usage:
    python train/dpo_smoke.py
"""

import os
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Week 4 SFT LoRA adapter (policy starts from here).
SFT_ADAPTER_DIR = Path("./outputs/sft-full")

# Week 3 preference set: {prompt_messages, chosen, rejected, ...}
DATASET_PATH = "data/processed/preferences_dpo.jsonl"
SMOKE_SIZE = 500
SPLIT_SEED = 42

# LoRA — same shape as Week 4 SFT so the adapter deltas compose cleanly.
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# DPO training. Smoke sizes only; the Week 6 full run will retune.
# A6000 48GB: keep per-device batch small since DPO holds chosen+rejected
# forward passes plus (implicit) reference forward passes in memory.
NUM_EPOCHS = 1
LEARNING_RATE = 5e-6                # conservative; DPO is loss-sensitive
DPO_BETA = 0.1                      # standard TRL default; ablate in Week 8
PER_DEVICE_BATCH = 2
GRAD_ACCUM_STEPS = 8                # effective batch = 2 * 8 = 16
MAX_LENGTH = 2048
MAX_PROMPT_LENGTH = 1536
WARMUP_RATIO = 0.03

# Cadence: with 500 pairs / eff-batch 16 = ~31 steps, so log/save often.
LOGGING_STEPS = 2
SAVE_STEPS = 25
SAVE_TOTAL_LIMIT = 2

OUTPUT_DIR = Path("./outputs/dpo-smoke")
WANDB_PROJECT = "llama-tools"
WANDB_RUN_NAME = "dpo-smoke-week5"


def load_preferences(path: str, tokenizer, n: int, seed: int):
    """Load preference JSONL and convert to DPOTrainer's expected schema.

    Source schema (from data/assemble_preferences.py):
        {prompt_messages: [{role, content}, ...], chosen: str, rejected: str, ...}

    DPOTrainer expects columns: {prompt: str, chosen: str, rejected: str}.
    We render `prompt` via the tokenizer's chat template with
    `add_generation_prompt=True` so the assistant turn boundary matches
    what `chosen` / `rejected` are the continuation of.
    """
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    def to_dpo(row):
        prompt = tokenizer.apply_chat_template(
            row["prompt_messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt,
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }

    keep = {"prompt", "chosen", "rejected"}
    drop = [c for c in ds.column_names if c not in keep]
    ds = ds.map(to_dpo, remove_columns=drop)
    return ds


def main() -> None:
    load_dotenv()

    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {SMOKE_SIZE} preference pairs from: {DATASET_PATH}")
    train_ds = load_preferences(DATASET_PATH, tokenizer, SMOKE_SIZE, SPLIT_SEED)
    print(f"  train: {len(train_ds)} pairs")
    print(f"  sample keys: {list(train_ds[0].keys())}")

    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Attaching Week 4 SFT adapter from: {SFT_ADAPTER_DIR}")
    # `is_trainable=True` so DPO can update these LoRA weights. The reference
    # forward pass is the same model with the adapter disabled (TRL handles
    # this automatically when ref_model=None and the policy is a PeftModel).
    policy = PeftModel.from_pretrained(
        base_model,
        str(SFT_ADAPTER_DIR),
        is_trainable=True,
    )

    # New LoRA config is NOT passed to DPOTrainer — we're continuing to train
    # the adapter that was just loaded, not attaching a fresh one on top.
    _ = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        # Output + logging
        output_dir=str(OUTPUT_DIR),
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name=WANDB_RUN_NAME,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        # Training
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        bf16=True,
        # Memory
        gradient_checkpointing=True,
        # DPO-specific
        beta=DPO_BETA,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        # Reproducibility
        seed=SPLIT_SEED,
    )

    # ref_model=None + PEFT policy => TRL uses adapter-disabled base as ref.
    # This is the memory-efficient pattern for a 48GB card.
    trainer = DPOTrainer(
        model=policy,
        ref_model=None,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    print("Starting DPO smoke training...")
    trainer.train()
    print("Training complete.")

    trainer.save_model(str(OUTPUT_DIR))
    trainer.save_state()
    print(f"Adapter + trainer state saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
