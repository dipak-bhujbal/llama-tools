# llama-tools

Post-trained Llama-3.1-8B for structured tool-calling in AI agents.

## What this is

An open-weight fine-tune of `meta-llama/Llama-3.1-8B-Instruct` optimized for reliable function-calling behavior: correct tool selection, correct argument formatting, correct types, no hallucinated tools. Full SFT + DPO training pipeline, evaluated on the Berkeley Function Calling Leaderboard (BFCL), with quantized variants for production serving.

## Status — 2026-07-20

**SFT stage: shipped.** `centuriandip/llama-3.1-8b-tools-sft` is live on HuggingFace (private; public flip planned after Week 7 eval). LoRA-SFT of Llama-3.1-8B-Instruct on 12,160 curated tool-calling examples, 3 epochs, 9h 09m on 1x RTX A6000, approximately $4.55 compute. Eval loss improved at all 11 checkpoints (0.4625 → 0.2117), zero overfitting signal through 3 full epochs. Released same-day via a signed, eval-gated checklist using the companion [release-kit](https://github.com/dipak-bhujbal/release-kit) framework.

**DPO v1: closed as a documented negative result.** Full DPO on the 10,242-pair rule-perturbed preference set degraded the policy rather than improving it. The pre-registered health metric (`eval_rewards/chosen`) caught the degradation mid-run (−0.36 → −0.53), the run was stopped early at step 400/622, and a checkpoint sweep with human diff-read confirmed monotonic regression — the SFT baseline beat every DPO checkpoint on the trainer's own holdout (16/20 vs 15/14/14/11 exact match; checkpoint-400 emitted invalid JSON). Root cause: rule-based rejecteds were trivially separable, so the optimizer had no ranking work left and ate the SFT signal instead. **The SFT model remains the shipped model.** Full analysis: [ADR-006](./docs/decisions/ADR-006-dpo-v1-negative-result.md).

| candidate | exact match | json valid |
|---|---|---|
| **sft (shipped)** | **16/20** | **20/20** |
| dpo-100 | 15/20 | 20/20 |
| dpo-200 | 14/20 | 20/20 |
| dpo-300 | 14/20 | 20/20 |
| dpo-400 | 11/20 | 18/20 |

**DPO v2: designed, optional scope.** Rejecteds sampled from the SFT model's own failures (on-policy hard negatives) instead of rule perturbations, with tightened kill lines and pre-registered abort conditions at every stage. Design: [ADR-007](./docs/decisions/ADR-007-dpo-v2-on-policy-rejecteds.md). Pipeline code is in `data/sample_failures.py`, `data/build_dpo_v2_pairs.py`, `train/dpo_v2_full.py`.

**Upcoming:** BFCL evaluation (upstream moved to v4; we will evaluate on the current leaderboard version), quantized variants, public model release.

See [`docs/progress/week-6-7-dpo-run-log.md`](./docs/progress/week-6-7-dpo-run-log.md) for the DPO arc lab notebook.

## Quick links

| Artifact | Location | State |
|---|---|---|
| SFT model | `centuriandip/llama-3.1-8b-tools-sft` | Private; public after eval |
| Preference dataset | `centuriandip/tool-calling-preferences` | Private until release |
| Final model (SFT+DPO) | `centuriandip/llama-3.1-8b-tools` | On hold — only ships if DPO v2 beats SFT |
| DPO v1 checkpoints + sweep report | HF staging (`dpo-checkpoints/`, `dpo-sweep/`) | Archived evidence (ADR-006) |
| Technical report | [`docs/report/`](./docs/report/) | In progress |
| ADRs | [`docs/decisions/`](./docs/decisions/) | 7 accepted |
| Lab notebook | [`docs/progress/week-6-7-dpo-run-log.md`](./docs/progress/week-6-7-dpo-run-log.md) | Live |

## Training results (SFT)

| Metric | Value |
|---|---|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| Training method | LoRA-SFT (r=64, alpha=128, merged) |
| Training examples | 12,160 (deduped from Hermes + xLAM) |
| Epochs | 3 |
| Hardware | 1x RTX A6000 48GB (Runpod) |
| Wall-clock / cost | 9h 09m / ~$4.55 |
| Eval loss (start → end) | 0.4625 → 0.2117 |
| Eval token accuracy (final) | 0.9445 |
| Overfit signal | None — improved at all 11 eval checkpoints |

Qualitative gate (5 held-out prompts, reviewed before release): 4/5 exact match to gold including three multi-call cases; 1/5 subtle argument miss (`"lr": "en-US"` vs expected `"pt-BR"` — right tool, right schema, wrong locale grounding). This argument-grounding failure class is the explicit target of the DPO stage.

BFCL benchmark numbers are not yet available. Formal evaluation against the current leaderboard version is scheduled for Week 7, with scores for base Llama-3.1-8B-Instruct and the shipped SFT model to be published at `eval/results/week-7.md` (plus DPO v2 if it earns a release per ADR-007).

## Data quality

Data quality is a first-class concern in this project, not an afterthought.

**Leakage prevention.** Zero overlap between the SFT/DPO training data and the BFCL eval set, verified via MinHash + exact match deduplication.

**Mid-run audit and bug catch.** A 200-pair spot-check during the Week 4 SFT run (model-assisted triage, human adjudication of all flags) surfaced an upstream xLAM annotation bug: argument values written as Python expressions (`"[0.02] * 5"`) rather than literal JSON arrays. Full datasets were searched: 16 SFT training targets and 15 DPO pairs affected (0.14% of the SFT set). Impact was quantified, an evidence-based decision was made not to restart the already-running job (0.14% cannot shift an 8B model meaningfully over 3 epochs), both datasets were cleaned, and the decision is recorded in the lab notebook with full rationale. The DPO set regenerated to 10,242 final pairs; SFT v2 cleaned to 12,143 (queued for future re-runs).

**Preference-set quality audit.** Human adjudication of the 200-pair sample: 194/200 OK, 2 bad-chosen (the xLAM bug), 3 trivial (already caught by the automated filter — the audit validated the filter, not just the pairs), 1 excluded for content. All decisions on record.

Both incidents are documented in [`docs/progress/week-4-run-log.md`](./docs/progress/week-4-run-log.md) and the relevant ADRs.

## What's in this repo

```
llama-tools/
├── PLAN.md                        # 12-week execution plan
├── ARCHITECTURE.md                # Technical design
├── data/                          # Data curation scripts (assembly, dedup, filtering)
├── train/                         # SFT and DPO training scripts (TRL-based)
├── eval/                          # BFCL eval harness + MMLU regression check
├── quantize/                      # AWQ int4 quantization pipeline
├── model_card/                    # Model card source (sft_model_card.md)
└── docs/
    ├── decisions/                 # 5 Architecture Decision Records (ADRs)
    ├── learning/                  # Mode A learning ramps (Weeks 1-4+)
    ├── progress/                  # Lab notebooks (weeks-1-3.md, week-4-run-log.md)
    └── report/                    # Technical report (in progress)
```

## Reproducing

The SFT training script is `train/sft_full.py`. Config as run: LoRA r=64 / alpha=128 / dropout 0.05 targeting q/k/v/o projections, 3 epochs, lr 2e-4 cosine with 3% warmup, effective batch 32, max sequence length 2048, bf16, gradient checkpointing. Full provenance (base model revision, dataset commit, random seed, hardware spec) is in [`docs/progress/week-4-run-log.md`](./docs/progress/week-4-run-log.md).

Complete end-to-end reproduction instructions will be published with the v1 release.

## Related projects

- **[release-kit](https://github.com/dipak-bhujbal/release-kit)** — open framework for eval-gated LLM releases. `llama-tools` is its reference implementation; the SFT release was the same-day first use of the signed checklist.

## License

Apache-2.0 (repo and scripts). Model weights: Meta Llama 3.1 Community License. Training data: Hermes (Apache-2.0), xLAM (CC-BY-4.0).

## Author

[Dipak Bhujbal](https://github.com/dipak-bhujbal)
