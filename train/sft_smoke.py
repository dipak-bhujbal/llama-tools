"""Week 2 smoke test: LoRA-SFT of Llama-3.1-8B-Instruct on 500 examples.

Purpose
-------
Verify the full training pipeline works end-to-end on real cloud GPU
hardware before committing to a $60-100 Week 4 full run. If this smoke
succeeds cleanly, Week 4's real SFT run is a scaled-up version of the
same script.

What this script does
---------------------
1. Loads Llama-3.1-8B-Instruct from HF (requires accepted license)
2. Loads NousResearch/hermes-function-calling-v1, takes 500 examples
3. Applies Llama-3.1 chat template to format examples
4. Attaches a LoRA adapter (rank 64, targets attention projections)
5. Trains for 1 epoch using TRL's SFTTrainer
6. Saves the adapter + optionally pushes to HF as a private repo
7. Logs training curves to wandb

Where to run this
-----------------
Runpod pod: 1x A100 40GB, Community Cloud → Spot (~$0.44/hr).
Expected wall-clock: 30-60 minutes.
Expected cost: $0.25-0.50.

Prerequisites (on the pod)
--------------------------
- Repo cloned + `pip install -e ".[train]"` completed
- `hf auth login --token $HF_TOKEN` ran successfully
- `wandb login $WANDB_API_KEY` ran successfully
- Llama-3.1-8B-Instruct license accepted on HF (approved)

Usage
-----
    python train/sft_smoke.py

Success signals (in wandb + terminal output)
--------------------------------------------
- Train loss curve: starts around 2.0-3.0, decreases smoothly, ends around 0.5-1.5
- Eval loss (on 50 held-out): decreases too, roughly tracks train loss
- No NaN / inf / OOM errors
- Final adapter saved to ./outputs/sft-smoke/
- Adapter pushed to huggingface.co/centuriandip/llama-3.1-8b-tools-week2-smoke (PRIVATE)

If something breaks, check docs/learning/week-2-sft-fundamentals.md
under "Failure modes and what they mean" for the common ones.
"""

import os
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# --------------------------------------------------------------------------
# Configuration — everything that might be tweaked lives up here.
# --------------------------------------------------------------------------

# Base model — see docs/decisions/ADR-001-base-model-selection.md.
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Function-calling dataset. Hermes format is already chat-templated
# messages, which saves us format-conversion work. For Week 4 we'll expand
# to Glaive + APIGen + xLAM per PLAN.md Week 3.
DATASET_ID = "NousResearch/hermes-function-calling-v1"
DATASET_CONFIG = "func_calling"  # sub-dataset within the repo
DATASET_SLICE = 500              # keep small for smoke test
EVAL_SPLIT = 50                  # held-out for eval loss

# LoRA config. Standard starting point for 7-8B models:
# - rank 64 balances capacity vs. cost
# - alpha 2x rank per the heuristic in the LoRA paper
# - target the attention projections; Llama-3 uses these standard names
# See docs/decisions/ADR-002-training-method-sft-dpo.md.
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training hyperparameters. Smoke test is 1 epoch on 500 examples so the
# whole run finishes quickly. Week 4 will use 3 epochs on 15K examples.
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4              # LoRA typical; ~10-100x higher than full FT
PER_DEVICE_BATCH = 4              # safe for A100 40GB + 8B + LoRA + 2048 seqs
GRAD_ACCUM_STEPS = 4              # effective batch = 4 * 4 = 16
MAX_SEQ_LENGTH = 2048             # covers most Hermes examples with headroom
WARMUP_RATIO = 0.03               # 3% warmup, then cosine decay

# Output + release
OUTPUT_DIR = Path("./outputs/sft-smoke")
HF_REPO_ID = "centuriandip/llama-3.1-8b-tools-week2-smoke"
PUSH_TO_HF = True   # set False to skip HF push
HF_PRIVATE = True   # smoke-test artifact stays private

# wandb
WANDB_PROJECT = "llama-tools"
WANDB_RUN_NAME = "sft-smoke-week2"


def main() -> None:
    # Load .env (HF_TOKEN, WANDB_API_KEY). Silent if file missing —
    # on Runpod the vars will typically already be exported in the shell.
    load_dotenv()

    # -----------------------------------------------------------------------
    # 1. Load the dataset.
    # -----------------------------------------------------------------------
    # `datasets.load_dataset` pulls from the HF Hub and caches locally.
    # We take a small slice and split into train / eval so we can measure
    # generalization even on a tiny dataset.
    print(f"Loading dataset: {DATASET_ID} ({DATASET_CONFIG})")
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")
    ds = ds.shuffle(seed=42).select(range(DATASET_SLICE))
    split = ds.train_test_split(test_size=EVAL_SPLIT, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  train: {len(train_ds)} examples, eval: {len(eval_ds)}")

    # -----------------------------------------------------------------------
    # 2. Load the tokenizer + model.
    # -----------------------------------------------------------------------
    # bf16 on A100 gives us significant memory savings + throughput vs. fp32.
    # `device_map="auto"` places layers on available GPU(s); for single-GPU
    # A100 that's just "all layers on cuda:0".
    print(f"Loading tokenizer + model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        # Llama's tokenizer has no pad token by default. Reusing EOS as pad
        # is standard practice for SFT — TRL masks pad tokens in loss.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # -----------------------------------------------------------------------
    # 3. Set up the LoRA adapter.
    # -----------------------------------------------------------------------
    # SFTTrainer with a peft_config will wrap the model with a LoRA adapter
    # automatically and freeze base weights. We just declare the config.
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # -----------------------------------------------------------------------
    # 4. Set up training arguments (SFTConfig subclasses TrainingArguments).
    # -----------------------------------------------------------------------
    training_args = SFTConfig(
        # Output + logging
        output_dir=str(OUTPUT_DIR),
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name=WANDB_RUN_NAME,
        logging_steps=5,
        save_strategy="epoch",
        # Eval
        eval_strategy="epoch",
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
        gradient_checkpointing=True,   # trades compute for memory; needed for 8B+LoRA
        # SFTConfig-specific
        max_length=MAX_SEQ_LENGTH,
        packing=False,                  # simple 1-example-per-sequence
        dataset_text_field="messages",  # will be auto chat-template-formatted
    )

    # -----------------------------------------------------------------------
    # 5. Create the trainer and train.
    # -----------------------------------------------------------------------
    # TRL will handle: chat template application, loss masking on prompt
    # tokens, LoRA wrapping, checkpointing. We just hand it the pieces.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # -----------------------------------------------------------------------
    # 6. Save + push.
    # -----------------------------------------------------------------------
    # trainer.save_model saves just the LoRA adapter (~200MB) — the base
    # model is downloaded fresh at load-time from HF, so we don't duplicate.
    trainer.save_model(str(OUTPUT_DIR))
    print(f"Adapter saved to: {OUTPUT_DIR}")

    if PUSH_TO_HF:
        print(f"Pushing adapter to HF: {HF_REPO_ID} (private={HF_PRIVATE})")
        trainer.push_to_hub(
            repo_id=HF_REPO_ID,
            private=HF_PRIVATE,
        )
        print("Pushed. Verify at: https://huggingface.co/" + HF_REPO_ID)


if __name__ == "__main__":
    main()
