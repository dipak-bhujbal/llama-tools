# October 2026 owner-confirmed project summary

Updated: 2026-10-08.

This is the current project summary confirmed by Dipak Bhujbal in October 2026, using his current Core AI Product resume as the reporting source. It is a versioned owner-reported summary. The historical Study 1 reports, per-item generations, answer-key comparison, and content-addressed evidence manifest remain separate records.

## Tool-calling post-training

| Reported measure | October 2026 summary |
|---|---|
| Model and method | Llama 3.1 8B; LoRA supervised fine-tuning, rank 64 |
| Curated examples | 12,143 |
| BFCL `simple_python` accuracy | 81.8% before post-training; 91.5% after |
| Argument errors | 73 to 34 out of 400 cases; 53% relative reduction |
| Function selection | At least 98.5% for both compared models |
| General-capability guardrail | MMLU remained inside the preset band |
| GPU time cost | $4.55 |
| Evaluation loss | 0.4625 to 0.2117 across 11 checkpoints |
| Training time and hardware | 9 hours on one A6000 |
| Evaluation harness | 687 passing tests |

The reported gain concerns argument correctness. Function selection was already high in both compared models. This is a result for the named BFCL slice, not a claim about the complete BFCL leaderboard or general agent reliability.

## Preference tuning and the no-go decision

The October summary reports two DPO arms, using 10,242 and 2,523 preference pairs, stopped against abort criteria declared before the runs and documented as negative results. The decision to retain SFT, rather than ship a preference-tuned checkpoint without demonstrated utility, is part of the project outcome.

The historical mechanisms and evaluation records are documented in [ADR-006](../decisions/ADR-006-dpo-v1-negative-result.md) and [ADR-008](../decisions/ADR-008-dpo-v2-negative-result.md). Their original results and statistical qualifications remain intact.

## Data, benchmark, and infrastructure integrity

- The upstream training-data defect affected 16 SFT targets and 15 DPO pairs: argument values were stored as code rather than literal data.
- A BFCL answer-key inconsistency was adjudicated at one row and filed upstream as [Gorilla issue #1354](https://github.com/ShishirPatil/gorilla/issues/1354). Paired comparisons held under both keys.
- The October summary reports a CUDA fault-isolation ladder running in 61 seconds, per-step telemetry, a liveness monitor, and a mandatory preflight gate.
- A full evaluation probe is reported to run in about 20 minutes for under one dollar.

## Evidence versions and availability

The [frozen Study 1 report](../../eval/results/study1_bfcl_simple_report.md) reports SFT 369/400 and DPO checkpoints 364/400, 363/400, and 359/400. Its [evidence manifest](../../eval/results/evidence.json) pins the public Hugging Face snapshot at revision `a3905a7381fd8bcbb16a04081cad595da6c7e616`. Those are the historical Study 1 scores and are not relabeled as the October comparison above.

The [public Hugging Face evidence bundle](https://huggingface.co/datasets/centuriandip/llama-3.1-8b-tools-dpo-v2-evidence/tree/a3905a7381fd8bcbb16a04081cad595da6c7e616) contains archived DPO checkpoints and historical evaluation artifacts. This documentation update does not add October raw runs, new model weights, or new training datasets. It does not claim that the frozen bundle independently reproduces the October figures.

The selected final SFT model and preference dataset remain private. The historical full-run MMLU raw-output gap remains documented in the evidence manifest; the public MMLU artifact is a 200-item smoke test. No new numerical MMLU band or full-run output is asserted here.
