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

## Week 5 runbook: DPO smoke test on a Runpod pod

### Pod

Same 1x RTX A6000 48GB as Week 4. ~40GB disk (base model + Week 4 adapter + small smoke output).

### Setup

Week 4 must have completed and `outputs/sft-full/` must be present on the pod (either the same pod, or `scp` the adapter dir up from wherever it lives). Also copy the preference set:

```bash
# from local machine
scp -P <pod-ssh-port> data/processed/preferences_dpo.jsonl root@<pod-ip>:/workspace/llama-tools/data/processed/
```

### Run

```bash
python train/dpo_smoke.py
```

Expected duration: **~30 min** (500 pairs, 1 epoch, ~31 optimizer steps at effective batch 16). Logs every 2 steps.

Success signals (all visible in the TRL step logs / wandb):
- `rewards/accuracies` climbing above **0.5** (chance) and trending toward 0.6+ by the end.
- `rewards/margins` growing (chosen reward pulling above rejected reward).
- `loss` dropping below the **0.693** baseline (–log 0.5 = the loss if the policy were indifferent between chosen and rejected).

If those three don't move in the right direction on 500 pairs, do NOT proceed to the Week 6 full run — debug the data or the ref-model wiring first.

## Week 6 runbook: full DPO on a Runpod pod

### Pod

Same 1x RTX A6000 48GB as Weeks 4-5. Pod setup is identical to the Week 5 smoke runbook (SFT adapter at `outputs/sft-full/`, `preferences_dpo.jsonl` at `data/processed/`, `hf auth login`, optional `wandb login`).

### Run

```bash
python train/dpo_full.py
```

Expected duration: **~8-10 hours** (~621 optimizer steps at the smoke's observed ~17 s/step; 9,942 train pairs / eff-batch 16 × 1 epoch). Logs every 10 steps, evals every 50, checkpoints every 100 (last 3 kept). Run under `tmux` so an SSH drop doesn't kill it.

### If the pod dies mid-run

Checkpoints persist in `outputs/dpo-full/checkpoint-*`. Resume with:

```bash
python train/dpo_full.py --resume
```

### Health checks (smoke-informed — read the module docstring for context)

- `rewards/accuracies` pinned at 1.0 is EXPECTED for this data (rule-based rejecteds are trivial for the SFT policy to separate). Do not read it as "converged".
- `rewards/chosen` must NOT go strongly negative — that is policy degradation.
- `eval_loss` should settle well below **0.693** (the indifference baseline) without crashing to ~0 (blown-up ref/policy gap / preference memorisation).
- The script prints a final metrics summary (last train + last eval log entry) after training so you can eyeball these without opening wandb.

### Post-run

Upload `outputs/dpo-full/` to HF staging BEFORE stopping the pod — the adapter is not in git and vanishes with the pod otherwise:

```bash
# from the pod (same staging dataset repo as the smoke archive)
hf upload centuriandip/llama-tools-sft-data outputs/dpo-full dpo-full --repo-type dataset
```

Week 8 will handle the merge + public push (analog of `merge_and_push.py`).

## Not in v1

- Full-parameter fine-tuning (LoRA only; full-param defers to a stretch goal)
- Multi-node distributed training (single-GPU sufficient for 8B)
- RLHF with a learned reward model (DPO only per ADR-002)

## Related

- `../data/` — datasets consumed here
- `../eval/` — where checkpoints are evaluated
- `../docs/decisions/ADR-002-training-method-sft-dpo.md` — method rationale
