# Architecture

This document describes the technical design of `llama-tools`. Decisions with meaningful alternatives are captured as ADRs under [`docs/decisions/`](./docs/decisions/).

## Goal

Produce an open-weight 8B model that measurably outperforms `Llama-3.1-8B-Instruct` on structured tool-calling (function-calling accuracy, argument formatting, tool selection), evaluated on the Berkeley Function Calling Leaderboard v3 (BFCL v3), while preserving general capability (MMLU regression within 2 points).

## Non-goals (v1)

- Multi-language function-calling (English-only for v1)
- Vision or multimodal tool-calling
- Model sizes other than 8B
- RLHF with a learned reward model (DPO is offline preference optimization; sufficient for v1)
- Custom kernel work (deferred; potential v2 pursuit as a separate resume artifact)

## Base model

**`meta-llama/Llama-3.1-8B-Instruct`** — see [ADR-001](./docs/decisions/ADR-001-base-model-selection.md) for the selection rationale (ecosystem gravity, weaker tool-calling baseline provides real headroom for improvement, well-supported by TRL/vLLM/AutoAWQ).

## Task and evaluation

**Primary task:** structured tool-calling — given a user query and a list of available functions (with JSON schemas), produce a correctly-formatted tool call or sequence of tool calls.

**Primary benchmark:** [BFCL v3](https://gorilla.cs.berkeley.edu/leaderboard.html) — a public, well-established leaderboard for function-calling accuracy. Four categories:

1. **Simple** — single function call with basic types
2. **Parallel** — multiple independent calls in one response
3. **Multiple** — chained calls where one call's output feeds another
4. **Multi-turn** — multi-step tool use across conversation turns

Score reported: accuracy per category + overall weighted score.

**Regression check:** [MMLU](https://github.com/hendrycks/test) — general knowledge benchmark used only as a *guardrail* to detect capability collapse. Target: within 2 points of base model.

**Success criteria:** meaningful BFCL uplift (target: +5-10 points overall vs base) while MMLU stays within 2 points. Numbers reported honestly in the model card whether or not the target is hit.

## Training pipeline

Two-stage post-training — see [ADR-002](./docs/decisions/ADR-002-training-method-sft-dpo.md) for method selection.

### Stage 1: Supervised Fine-Tuning (SFT)

- **Framework:** [TRL](https://huggingface.co/docs/trl) `SFTTrainer`
- **Data:** ~15K high-quality function-calling examples curated from Glaive function-calling, APIGen, xLAM function-calling-60k
- **Format:** conversation-style with Llama-3.1 chat template, tool-call responses as structured JSON in the assistant turn
- **LoRA vs full fine-tune:** LoRA (rank 64, alpha 128) for cost efficiency; ablation vs full fine-tune deferred to a stretch goal
- **Hyperparameters:** starting point — 3 epochs, lr 2e-4 (LoRA), batch 8, sequence length 4096, cosine schedule, warmup 3%
- **Compute:** ~1x A100 40GB × 6-8 hours per run

### Stage 2: Direct Preference Optimization (DPO)

- **Framework:** TRL `DPOTrainer`
- **Data:** ~15K preference pairs where "chosen" = correct tool call, "rejected" = plausible-but-wrong tool call (wrong function, missing argument, malformed JSON, hallucinated tool). Rejected examples synthesized by adversarial perturbation of chosen examples plus real-world failure patterns.
- **Reference model:** the SFT checkpoint from Stage 1
- **Hyperparameters:** starting point — beta 0.1, 3 epochs, lr 5e-7 (LoRA), batch 4. Beta ablation in Week 8 (0.1, 0.3, 0.5).
- **Compute:** ~1x A100 40GB × 6-10 hours per run

## Quantization

**AWQ int4** via [AutoAWQ](https://github.com/casper-hansen/AutoAWQ). Calibration set drawn from held-out tool-calling data. Published as a separate HF repo variant.

**Target:** <2% BFCL degradation vs full-precision, ~3-4x memory reduction, ~1.5-2x throughput improvement on vLLM.

## Serving benchmark

Post-release: benchmark on [vLLM](https://github.com/vllm-project/vllm) — tokens/sec throughput, first-token latency, throughput under concurrent load. Report all numbers in the model card and technical report.

## Compute budget

Total cloud budget: **$1000 hard cap**. Estimated allocation:

| Category | Estimated spend |
|---|---|
| Learning + smoke tests (Weeks 1-2) | $20-35 |
| Data prep + tokenization (Week 3) | $5 |
| SFT full runs (Week 4) | $60-100 |
| DPO full runs (Week 6) | $80-120 |
| Evaluation runs (Week 7) | $40-60 |
| Ablations (Week 8) | $120-180 |
| Quantization + benchmark (Weeks 9, 11) | $30-50 |
| Buffer for reruns + debugging | $400+ |

Provider: **Runpod community spot** (A100 40GB @ ~$0.44/hr) as primary; Lambda Labs as fallback if Runpod ergonomics prove painful.

## Publishing artifacts

Every v1 deliverable is a permanent inspectable artifact:

1. **`centuriandip/llama-3.1-8b-tools`** — full-precision merged model (LoRA merged into base for ease of use)
2. **`centuriandip/llama-3.1-8b-tools-awq`** — AWQ int4 variant
3. **`centuriandip/tool-calling-preferences`** — the curated preference dataset (HF dataset repo)
4. **[`docs/technical-report.md`](./docs/technical-report.md)** — ~2000-word writeup: methodology, results, ablations, honest failure analysis, cost breakdown
5. **Model cards** on both HF model repos — proper structure per [HF model card guidelines](https://huggingface.co/docs/hub/model-cards)
