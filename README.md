# llama-tools

Post-trained Llama-3.1-8B for structured tool-calling in AI agents.

## What this is

An open-weight fine-tune of `meta-llama/Llama-3.1-8B-Instruct` optimized for reliable function-calling behavior: correct tool selection, correct argument formatting, correct types, no hallucinated tools. The repository contains the SFT and DPO training pipelines, BFCL and MMLU evaluation code, and the evidence behind the decision to ship SFT without either DPO variant. Quantized variants remain planned work.

## Status — 2026-08-04

**SFT is the selected final model.** `centuriandip/llama-3.1-8b-tools-sft` remains private on Hugging Face. LoRA-SFT on 12,160 curated tool-calling examples ran for 3 epochs (9h 09m on 1x RTX A6000, approximately $4.55 compute). Eval loss improved at all 11 checkpoints (0.4625 → 0.2117). On the 400-item BFCL v4 `simple_python` set, SFT scored **369/400 = 92.25%** with perfect function-name selection and JSON validity. The accepted study-1 record reports MMLU 5-shot **0.659 vs 0.683 base** (−2.4 points).

**DPO v1: closed as a documented negative result.** Full DPO on the 10,242-pair rule-perturbed preference set degraded the policy rather than improving it. The pre-registered health metric (`eval_rewards/chosen`) caught the degradation mid-run (−0.36 → −0.53), the run was stopped early at step 400/622, and a checkpoint sweep with human diff-read confirmed monotonic regression — the SFT baseline beat every DPO checkpoint on the trainer's own holdout (16/20 vs 15/14/14/11 exact match; checkpoint-400 emitted invalid JSON). Root cause: rule-based rejecteds were trivially separable, so the optimizer had no ranking work left and ate the SFT signal instead. **The SFT model remains the shipped model.** Full analysis: [ADR-006](./docs/decisions/ADR-006-dpo-v1-negative-result.md).

| candidate | exact match | json valid |
|---|---|---|
| **sft (shipped)** | **16/20** | **20/20** |
| dpo-100 | 15/20 | 20/20 |
| dpo-200 | 14/20 | 20/20 |
| dpo-300 | 14/20 | 20/20 |
| dpo-400 | 11/20 | 18/20 |

**DPO v2: closed as a second documented negative result.** The on-policy hard-negative design fixed DPO v1's training-time failure: held-out preference accuracy and margins improved without runaway chosen-reward damage. Held-out tool-use was still lower than SFT at every evaluated DPO checkpoint — **364/400 at step 50, 363/400 at step 100, and 359/400 at step 150**, against SFT's 369/400 — and no paired contrast was significant after Holm correction. The training signal did not transfer to fresh ground-truth tool use, so no DPO checkpoint shipped. Full analysis: [ADR-008](./docs/decisions/ADR-008-dpo-v2-negative-result.md).

**Study 2: preparation only.** A follow-up study will test on-policy mining under bounded objectives and ground-truth checkpoint selection. Baselines, data provenance, endpoint qualification, and paid stages are gated before execution. No study-2 model run has started.

The BFCL report and per-item generations are preserved in the public [`llama-3.1-8b-tools-dpo-v2-evidence`](https://huggingface.co/datasets/centuriandip/llama-3.1-8b-tools-dpo-v2-evidence) dataset and indexed under [`eval/results/`](./eval/results/). The public MMLU file is the 200-item smoke; the full 14,042-item raw output has not yet been recovered, and this evidence gap is recorded explicitly rather than filled from memory.

## Quick links

| Artifact | Location | State |
|---|---|---|
| SFT model | `centuriandip/llama-3.1-8b-tools-sft` | Private; selected final model |
| Preference dataset | `centuriandip/tool-calling-preferences` | Private |
| Planned SFT+DPO model | `centuriandip/llama-3.1-8b-tools` | Not published; both DPO variants failed the ship gate |
| DPO v1 checkpoints + sweep report | HF staging (`dpo-checkpoints/`, `dpo-sweep/`) | Archived evidence (ADR-006) |
| DPO v2 checkpoints + eval evidence | [`centuriandip/llama-3.1-8b-tools-dpo-v2-evidence`](https://huggingface.co/datasets/centuriandip/llama-3.1-8b-tools-dpo-v2-evidence) | Public negative-result evidence |
| Committed eval index | [`eval/results/`](./eval/results/) | BFCL report + content-addressed evidence manifest |
| Technical report | [`docs/report/`](./docs/report/) | In progress |
| ADRs | [`docs/decisions/`](./docs/decisions/) | 8 accepted |
| Lab notebook | [`docs/progress/week-6-7-dpo-run-log.md`](./docs/progress/week-6-7-dpo-run-log.md) | Study-1 DPO arc closed |

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

Formal BFCL v4 `simple_python` results are committed at [`eval/results/study1_bfcl_simple_report.md`](./eval/results/study1_bfcl_simple_report.md). The frozen input/key manifest contains **400** rows and unique ids, matching the scorer and archived per-item generations. The earlier `n=399` text was a newline-counting error and has been corrected; the reported 369/400 score itself was always 92.25%.

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
    ├── decisions/                 # 8 accepted Architecture Decision Records (ADRs)
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
