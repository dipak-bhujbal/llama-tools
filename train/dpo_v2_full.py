"""DPO v2 full run: LoRA-DPO of the SFT policy on on-policy hard pairs.

Implements ADR-007 Stage 3. Identical pipeline to train/dpo_full.py (v1)
except for the deltas below — each traceable to a v1 lesson (ADR-006):

- Data: `preferences_dpo_v2.jsonl` (rejecteds sampled from the SFT model's
  own failures) instead of rule-perturbed pairs.
- MECHANICAL kill line: training auto-stops if `eval_rewards/chosen` drops
  below -0.25 (v1 showed visible policy damage by -0.36; v1's -0.6 line was
  too permissive and was enforced only by a human watching).
- MECHANICAL easy-data abort: if the FIRST eval reports
  `eval_rewards/accuracies` >= 0.99, the pairs are still too easy — the v2
  hypothesis is falsified and training stops immediately.
- `load_best_model_at_end=True` + `metric_for_best_model=eval_loss`
  (Week 4 run-log 15:43 lesson, missed again in v1).
- Finer cadence (eval 25 / save 50) for the smaller hard-pair dataset.
- Same epochs / LR / beta as v1 — isolate the data variable.

Prereqs on the pod: as train/dpo_full.py, plus
`data/processed/preferences_dpo_v2.jsonl` (Stage 2 gate must have PASSED).

Usage:
    python train/dpo_v2_full.py
    python train/dpo_v2_full.py --resume
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SFT_ADAPTER_DIR = Path("./outputs/sft-full")

DATASET_PATH = "data/processed/preferences_dpo_v2.jsonl"
EVAL_SPLIT_FRACTION = 0.05          # dataset size unknown until Stage 2 runs
MIN_EVAL_SPLIT = 100
SPLIT_SEED = 42

NUM_EPOCHS = 1
LEARNING_RATE = 5e-6
DPO_BETA = 0.1
PER_DEVICE_BATCH = 2
GRAD_ACCUM_STEPS = 8                # effective batch 16
MAX_LENGTH = 2048
WARMUP_RATIO = 0.03

EVAL_STEPS = 25
SAVE_STEPS = 50
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 4

CHOSEN_REWARD_KILL_LINE = -0.25     # ADR-007
EASY_DATA_ACCURACY_ABORT = 0.99     # ADR-007: pinned accuracies on 1st eval

OUTPUT_DIR = Path("./outputs/dpo-v2-full")
WANDB_PROJECT = "llama-tools"
WANDB_RUN_NAME = "dpo-v2-onpolicy"


class PreRegisteredAborts(TrainerCallback):
    """ADR-007 mechanical abort conditions — no human watching required."""

    def __init__(self):
        self.first_eval_seen = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        chosen = metrics.get("eval_rewards/chosen")
        acc = metrics.get("eval_rewards/accuracies")
        if not self.first_eval_seen:
            self.first_eval_seen = True
            if acc is not None and acc >= EASY_DATA_ACCURACY_ABORT:
                print(f"\nABORT (pre-registered): first-eval accuracies {acc:.3f} "
                      f">= {EASY_DATA_ACCURACY_ABORT} — pairs still too easy, "
                      "v2 hypothesis falsified. Stopping.")
                control.should_training_stop = True
                return
        if chosen is not None and chosen < CHOSEN_REWARD_KILL_LINE:
            print(f"\nKILL LINE (pre-registered): eval_rewards/chosen {chosen:.3f} "
                  f"< {CHOSEN_REWARD_KILL_LINE} — policy degradation. Stopping; "
                  "last good checkpoint is the sweep candidate.")
            control.should_training_stop = True


def load_preferences(path: str, tokenizer, seed: int):
    ds = load_dataset("json", data_files=path, split="train")

    def to_dpo(row):
        prompt = tokenizer.apply_chat_template(
            row["prompt_messages"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}

    keep = {"prompt", "chosen", "rejected"}
    drop = [c for c in ds.column_names if c not in keep]
    ds = ds.map(to_dpo, remove_columns=drop)

    eval_size = max(MIN_EVAL_SPLIT, int(len(ds) * EVAL_SPLIT_FRACTION))
    split = ds.train_test_split(test_size=eval_size, seed=seed)
    return split["train"], split["test"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_dotenv()

    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading preference pairs from: {DATASET_PATH}")
    train_ds, eval_ds = load_preferences(DATASET_PATH, tokenizer, SPLIT_SEED)
    print(f"  train: {len(train_ds)} pairs, eval: {len(eval_ds)} pairs")

    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto"
    )

    print(f"Attaching SFT adapter from: {SFT_ADAPTER_DIR}")
    policy = PeftModel.from_pretrained(
        base_model, str(SFT_ADAPTER_DIR), is_trainable=True
    )

    training_args = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name=WANDB_RUN_NAME,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        beta=DPO_BETA,
        max_length=MAX_LENGTH,
        seed=SPLIT_SEED,
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[PreRegisteredAborts()],
    )

    print("Starting DPO v2 training (on-policy hard pairs)...")
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    print("Training finished (completed or pre-registered stop).")

    trainer.save_model(str(OUTPUT_DIR))
    trainer.save_state()
    print(f"Adapter + trainer state saved to: {OUTPUT_DIR}")

    print("\n=== Final DPO metrics (last log entries) ===")
    history = trainer.state.log_history
    last_train = next(
        (e for e in reversed(history) if "loss" in e and "eval_loss" not in e), None
    )
    last_eval = next((e for e in reversed(history) if "eval_loss" in e), None)
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
        ]
        vals = {k: entry[k] for k in keys if k in entry}
        print(f"  {label}: {vals}")
    print("=== end summary ===")
    print("\nNext: python eval/dpo_sweep.py with v2 checkpoints "
          "(pass --checkpoints matching outputs/dpo-v2-full/checkpoint-*)")


if __name__ == "__main__":
    main()
