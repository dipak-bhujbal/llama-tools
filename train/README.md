# train/

Training scripts. Populated across **Weeks 2, 4, 6, 8**.

## What lands here

- **`sft_smoke.py`** — Week 2. LoRA-SFT on 500 examples. Verifies the training loop wires up. ~30-60 min on a rented GPU.
- **`sft_full.py`** — Week 4. Full LoRA-SFT of Llama-3.1-8B-Instruct on the 12,160-example curated set. 3 epochs, ~6-8 hours on 1x A6000 48GB.
- **`merge_and_push.py`** — Week 4. Merges the LoRA adapter into base weights and uploads the merged model to `centuriandip/llama-3.1-8b-tools-sft`.
- **`dpo_smoke.py`** / **`dpo.py`** / **`ablate_dpo_beta.py`** — Weeks 5, 6, 8 (not yet written).

## Week 4 runbook: full SFT on a Runpod pod

### Pod

1x RTX A6000 48GB (or A100 40GB — see hardware note below). ~60GB disk minimum (base model ~16GB + 3 checkpoints + merged output ~30GB).

### Setup

```bash
# on the pod
git clone https://github.com/dipak-bhujbal/llama-tools && cd llama-tools
pip install -e ".[train]"
hf auth login --token $HF_TOKEN
wandb login $WANDB_API_KEY   # optional; script falls back to plain logging

# the dataset is not in git — copy it up from local:
# (from local machine)
scp -P <pod-ssh-port> data/processed/sft_dedup.jsonl root@<pod-ip>:/workspace/llama-tools/data/processed/
```

### Run

```bash
python train/sft_full.py
```

Expected duration: **~6-8 hours** (~1090 optimizer steps). Eval loss logs every 100 steps; checkpoints every 200 steps (last 3 kept). Run it under `tmux` so an SSH drop doesn't kill it.

Success signals: train loss decreasing smoothly, eval loss tracking it without diverging upward (divergence past epoch 2 = overfitting; the epoch-2 checkpoint is still on disk if so). Final adapter + `trainer_state.json` + `sample_generations.json` land in `outputs/sft-full/`.

### If the pod dies mid-run

Checkpoints persist in `outputs/sft-full/checkpoint-*` (use a Runpod network volume, or re-`scp` the outputs dir off the pod periodically). To resume:

```bash
python train/sft_full.py --resume
```

This picks up from the latest checkpoint including optimizer/scheduler state.

### Merge + publish

```bash
python train/merge_and_push.py
```

Merges the adapter, saves the full bf16 model to `outputs/sft-full-merged/`, and uploads to `centuriandip/llama-3.1-8b-tools-sft` via `HfApi.upload_folder` (private first; flip public after Week 7 eval). Add `--skip-upload` to merge locally only.

### Hardware note

Defaults are tuned for A6000 48GB (`PER_DEVICE_BATCH=8`, `GRAD_ACCUM_STEPS=4`). On an A100 40GB, edit the config block to `PER_DEVICE_BATCH=4`, `GRAD_ACCUM_STEPS=8` — same effective batch of 32, so the loss curves stay comparable.

## Not in v1

- Full-parameter fine-tuning (LoRA only; full-param defers to a stretch goal)
- Multi-node distributed training (single-GPU sufficient for 8B)
- RLHF with a learned reward model (DPO only per ADR-002)

## Related

- `../data/` — datasets consumed here
- `../eval/` — where checkpoints are evaluated
- `../docs/decisions/ADR-002-training-method-sft-dpo.md` — method rationale
