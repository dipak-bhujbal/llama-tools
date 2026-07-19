"""Week 6 full DPO: LoRA-DPO of the Week 4 SFT policy on the 10,242 preference set.

Scale-up of `train/dpo_smoke.py`. Same pipeline, real data, 1 epoch (DPO
overtrains fast — PLAN.md Week 6). Target hardware: 1x RTX A6000 48GB (Runpod).
Expected wall-clock: ~8-10 hours at the smoke's observed ~17 s/step.

Smoke-informed watch note (from the Week 5 smoke run):
- `rewards/accuracies` pinned at 1.0 is EXPECTED: the rule-based rejecteds are
  trivially separable for an SFT-ed policy (logps/rejected ≈ -40 vs
  logps/chosen ≈ -0.4). Do not read 1.0 as "converged".
- Margins in the smoke were driven by pushing REJECTED down, with
  `rewards/chosen` hovering near 0 (not going negative). That is the healthy
  regime for this data.
- Full-run health checks (must hold throughout):
    * `rewards/chosen` must NOT go strongly negative → that is policy
      degradation (the DPO update is eating the SFT signal).
    * `eval_loss` should settle well below 0.693 (–log 0.5, the indifference
      baseline) WITHOUT crashing to ~0 (which would mean the ref/policy gap
      blew up and the model is memorising the preference set).

Prereqs on the pod:
- Week 4 SFT complete; `outputs/sft-full/` present with the LoRA adapter.
- `pip install -e ".[train]"` completed
- `hf auth login --token $HF_TOKEN` succeeded
- `wandb login $WANDB_API_KEY` (optional; script falls back to plain logging)
- `data/processed/preferences_dpo.jsonl` present on the pod (10,242 pairs)

Usage:
    python train/dpo_full.py
    # resume from last checkpoint if the pod died mid-run:
    python train/dpo_full.py --resume
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Week 4 SFT LoRA adapter (policy starts from here).
SFT_ADAPTER_DIR = Path("./outputs/sft-full")

# Week 3 full preference set: {prompt_messages, chosen, rejected, ...}
DATASET_PATH = "data/processed/preferences_dpo.jsonl"
EVAL_SPLIT = 300
SPLIT_SEED = 42

# DPO training. Same hyperparams as the smoke (proven-good defaults).
# per-device 2 × grad-accum 8 = effective 16, which fit A6000 48GB in the
# smoke. ~9942 train pairs / 16 eff-batch = ~621 optimizer steps @ 1 epoch.
NUM_EPOCHS = 1
LEARNING_RATE = 5e-6
DPO_BETA = 0.1
PER_DEVICE_BATCH = 2
GRAD_ACCUM_STEPS = 8               # effective batch = 2 * 8 = 16
# max_prompt_length was DROPPED from DPOConfig post commit d22ce8b in the
# TRL version on the pod — do not reintroduce it. max_length alone caps the
# joint prompt+completion sequence.
MAX_LENGTH = 2048
WARMUP_RATIO = 0.03

# Cadence
EVAL_STEPS = 50
SAVE_STEPS = 100
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 3

OUTPUT_DIR = Path("./outputs/dpo-full")
WANDB_PROJECT = "llama-tools"
WANDB_RUN_NAME = "dpo-full-week6"


def load_preferences(path: str, tokenizer, eval_size: int, seed: int):
    """Load full preference JSONL, convert to DPOTrainer schema, hold out eval.

    Source schema (from data/assemble_preferences.py):
        {prompt_messages: [{role, content}, ...], chosen: str, rejected: str, ...}

    DPOTrainer expects columns: {prompt: str, chosen: str, rejected: str}.
    We render `prompt` via the tokenizer's chat template with
    `add_generation_prompt=True` so the assistant turn boundary matches
    what `chosen` / `rejected` are the continuation of.
    """
    ds = load_dataset("json", data_files=path, split="train")

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

    split = ds.train_test_split(test_size=eval_size, seed=seed)
    return split["train"], split["test"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint in OUTPUT_DIR.",
    )
    args = parser.parse_args()

    load_dotenv()

    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading preference pairs from: {DATASET_PATH}")
    train_ds, eval_ds = load_preferences(
        DATASET_PATH, tokenizer, EVAL_SPLIT, SPLIT_SEED
    )
    print(f"  train: {len(train_ds)} pairs, eval: {len(eval_ds)} pairs")
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

    training_args = DPOConfig(
        # Output + logging
        output_dir=str(OUTPUT_DIR),
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name=WANDB_RUN_NAME,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        # Eval
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
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
        # Reproducibility
        seed=SPLIT_SEED,
    )

    # ref_model=None + PEFT policy => TRL uses adapter-disabled base as ref.
    trainer = DPOTrainer(
        model=policy,
        ref_model=None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("Starting DPO full training...")
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    print("Training complete.")

    trainer.save_model(str(OUTPUT_DIR))
    trainer.save_state()
    print(f"Adapter + trainer state saved to: {OUTPUT_DIR}")

    # Final health summary: pull the last log entry and print the DPO metrics
    # that matter (see module docstring for what "healthy" looks like).
    print("\n=== Final DPO metrics (last log entry) ===")
    history = trainer.state.log_history
    last_train = next(
        (e for e in reversed(history) if "loss" in e and "eval_loss" not in e),
        None,
    )
    last_eval = next(
        (e for e in reversed(history) if "eval_loss" in e), None
    )
    for label, entry in (("train", last_train), ("eval", last_eval)):
        if entry is None:
            print(f"  {label}: (no entry found)")
            continue
        keys = [
            "loss", "eval_loss",
            "rewards/chosen", "eval_rewards/chosen",
            "rewards/rejected", "eval_rewards/rejected",
            "rewards/accuracies", "eval_rewards/accuracies",
            "rewards/margins", "eval_rewards/margins",
            "logps/chosen", "eval_logps/chosen",
            "logps/rejected", "eval_logps/rejected",
        ]
        vals = {k: entry[k] for k in keys if k in entry}
        print(f"  {label}: {vals}")
    print("=== end summary ===")


if __name__ == "__main__":
    main()
