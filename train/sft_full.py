"""Week 4 full SFT: LoRA-SFT of Llama-3.1-8B-Instruct on the 12,160-example set.

Scale-up of `train/sft_smoke.py`. Same pipeline, real data, 3 epochs. Target
hardware: 1x RTX A6000 48GB (Runpod). Expected wall-clock: ~6-8 hours.

Prereqs on the pod:
- `pip install -e ".[train]"` completed
- `hf auth login --token $HF_TOKEN` succeeded
- `wandb login $WANDB_API_KEY` succeeded (optional; skipped if var absent)
- Llama-3.1-8B-Instruct license accepted on HF
- `data/processed/sft_dedup.jsonl` present on the pod (rsync it up before run)

Usage:
    python train/sft_full.py
    # resume from last checkpoint if the pod died mid-run:
    python train/sft_full.py --resume
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Week 3 deduped SFT set (12,160 examples: messages + source + source_id).
DATASET_PATH = "data/processed/sft_dedup.jsonl"
EVAL_SPLIT = 500
SPLIT_SEED = 42

# LoRA — Week 2 deferred defaults, now committed per PLAN.md Week 4.
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training. A6000 48GB has ~20% more headroom than the A100 40GB the smoke
# ran on, so per-device batch goes 4 -> 8; effective batch held at 32.
# If running on A100 40GB instead: PER_DEVICE_BATCH = 4, GRAD_ACCUM_STEPS = 8.
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH = 8
GRAD_ACCUM_STEPS = 4               # effective batch = 8 * 4 = 32
MAX_SEQ_LENGTH = 2048
WARMUP_RATIO = 0.03

# Checkpointing + eval cadence. ~1090 optimizer steps total
# (11660 train examples / 32 effective batch * 3 epochs).
EVAL_STEPS = 100
SAVE_STEPS = 200
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 3               # keep last 3 checkpoints on disk

OUTPUT_DIR = Path("./outputs/sft-full")
WANDB_PROJECT = "llama-tools"
WANDB_RUN_NAME = "sft-full-week4"

NUM_SAMPLE_GENERATIONS = 5


def load_local_sft(path: str):
    """Load the Week 3 JSONL. Each line: {messages, source, source_id}."""
    ds = load_dataset("json", data_files=path, split="train")
    split = ds.train_test_split(test_size=EVAL_SPLIT, seed=SPLIT_SEED)
    return split["train"], split["test"]


def generate_samples(model, tokenizer, eval_ds, out_path: Path, n: int) -> None:
    """Greedy-decode held-out prompts for a qualitative check."""
    model.eval()
    samples = []
    for i in range(min(n, len(eval_ds))):
        msgs = eval_ds[i]["messages"]
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        if not prompt_msgs:
            continue
        gold = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        samples.append({"prompt": prompt, "generated": gen, "gold": gold})
    out_path.write_text(json.dumps(samples, indent=2))
    print(f"Sample generations written to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint in OUTPUT_DIR.",
    )
    args = parser.parse_args()

    load_dotenv()

    print(f"Loading dataset: {DATASET_PATH}")
    train_ds, eval_ds = load_local_sft(DATASET_PATH)
    print(f"  train: {len(train_ds)} examples, eval: {len(eval_ds)}")

    print(f"Loading tokenizer + model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
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
        # SFTConfig-specific
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        dataset_text_field="messages",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    print("Training complete.")

    trainer.save_model(str(OUTPUT_DIR))
    trainer.save_state()
    print(f"Adapter + trainer state saved to: {OUTPUT_DIR}")

    generate_samples(
        trainer.model,
        tokenizer,
        eval_ds,
        OUTPUT_DIR / "sample_generations.json",
        n=NUM_SAMPLE_GENERATIONS,
    )


if __name__ == "__main__":
    main()
