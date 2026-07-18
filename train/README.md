# train/

Training scripts. Populated across **Weeks 2, 4, 6, 8**.

## What lands here

TRL-based training scripts for SFT and DPO, plus the ablation runners.

- **`sft_smoke.py`** — Week 2. LoRA-SFT on 500 examples. Verifies the training loop wires up. Should finish in ~30-60 minutes on a rented A100.
- **`sft.py`** — Week 4. Full LoRA-SFT of Llama-3.1-8B-Instruct on the curated 15K set. ~6-8 hours on 1x A100 40GB.
- **`dpo_smoke.py`** — Week 5. Tiny DPO run to verify the preference-training loop.
- **`dpo.py`** — Week 6. Full DPO run on the SFT checkpoint. ~8-10 hours.
- **`ablate_dpo_beta.py`** — Week 8. Runs DPO with beta ∈ {0.1, 0.3, 0.5}; parameterized wrapper around `dpo.py`.
- **`merge_lora.py`** — merges a LoRA adapter into the base model weights for HF publication.

## Configuration

Training configs live in `train/configs/` as YAML — one file per run so every experiment is reproducible from disk.

## Not in v1

- Full-parameter fine-tuning (LoRA only; full-param defers to a stretch goal)
- Multi-node distributed training (single-GPU sufficient for 8B)
- RLHF with a learned reward model (DPO only per ADR-002)

## Related

- `../data/` — datasets consumed here
- `../eval/` — where checkpoints are evaluated
- `../docs/decisions/ADR-002-training-method-sft-dpo.md` — method rationale
