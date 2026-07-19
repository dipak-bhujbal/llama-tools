"""Merge the Week 4 LoRA adapter into base weights and upload to HF.

Loads the base model in bf16, applies the trained adapter from
`outputs/sft-full/`, merges (merge_and_unload), saves the full merged model
locally, then uploads via HfApi.upload_folder. We deliberately avoid
trainer/model .push_to_hub — it broke under TRL/transformers version drift
in Week 2 (`create_model_card() got unexpected keyword argument 'repo_id'`);
HfApi.upload_folder is the stable path.

Merging needs ~16GB of RAM/VRAM for the bf16 8B weights plus ~30GB of disk
for the saved output. Run on the pod after training, or on any machine with
enough disk (device_map="auto" falls back to CPU fine — merging is not
compute-bound).

Usage:
    python train/merge_and_push.py
    python train/merge_and_push.py --skip-upload   # merge only
"""

import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_DIR = Path("./outputs/sft-full")
MERGED_DIR = Path("./outputs/sft-full-merged")

HF_REPO_ID = "centuriandip/llama-3.1-8b-tools-sft"
HF_PRIVATE = True   # flip to public after eval sanity-check (Week 7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-upload", action="store_true", help="Merge and save locally only."
    )
    args = parser.parse_args()

    load_dotenv()

    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Applying adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {MERGED_DIR}")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR), safe_serialization=True)

    # Ship the tokenizer alongside so the repo is loadable standalone.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(str(MERGED_DIR))

    if args.skip_upload:
        print("Skipping upload (--skip-upload).")
        return

    print(f"Uploading to HF: {HF_REPO_ID} (private={HF_PRIVATE})")
    api = HfApi()
    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type="model",
        private=HF_PRIVATE,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(MERGED_DIR),
        repo_id=HF_REPO_ID,
        repo_type="model",
    )
    print("Uploaded. Verify at: https://huggingface.co/" + HF_REPO_ID)


if __name__ == "__main__":
    main()
