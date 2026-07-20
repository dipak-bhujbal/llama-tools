"""Merge the winning DPO checkpoint into base weights and upload to HF.

Same pipeline as `merge_and_push.py` (Week 4 SFT merge) but parameterised on
the sweep-selected checkpoint. The DPO checkpoints are LoRA adapters trained
on top of the base — attaching the checkpoint adapter alone reproduces the
policy (the SFT adapter was the starting point of those weights, not a
separate layer to stack).

As in Week 4, upload goes through HfApi.upload_folder, never
trainer/model .push_to_hub (broke under TRL/transformers version drift).

Usage:
    python train/merge_dpo_winner.py --checkpoint 300
    python train/merge_dpo_winner.py --checkpoint 300 --skip-upload
"""

import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CHECKPOINT_ROOT = Path("./outputs/dpo-checkpoints")
MERGED_DIR = Path("./outputs/dpo-winner-merged")

HF_REPO_ID = "centuriandip/llama-3.1-8b-tools-dpo"
HF_PRIVATE = True   # flip to public after Week 7 eval sanity-check


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=int, required=True,
                        help="Winning checkpoint step, e.g. 300")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    load_dotenv()

    adapter_dir = CHECKPOINT_ROOT / f"checkpoint-{args.checkpoint}"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint: {adapter_dir}")

    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Loading base model: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto"
    )

    print(f"Attaching DPO adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to: {MERGED_DIR}")
    merged.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))

    # Keep the winning adapter + provenance alongside the merged weights.
    adapter_out = MERGED_DIR / "adapter"
    print(f"Copying winning adapter to: {adapter_out}")
    import shutil
    shutil.copytree(adapter_dir, adapter_out, dirs_exist_ok=True)

    if args.skip_upload:
        print("Skipping upload (--skip-upload).")
        return

    print(f"Uploading to HF: {HF_REPO_ID} (private={HF_PRIVATE})")
    api = HfApi()
    api.create_repo(HF_REPO_ID, private=HF_PRIVATE, exist_ok=True)
    api.upload_folder(
        repo_id=HF_REPO_ID,
        folder_path=str(MERGED_DIR),
        commit_message=f"DPO winner: checkpoint-{args.checkpoint} merged into {BASE_MODEL}",
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
