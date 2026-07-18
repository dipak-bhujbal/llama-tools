# eval/

Evaluation harness. Populated in **Week 7** (with the BFCL v3 harness as the first serious component; will also be reused inside `release-kit`).

## What lands here

- **`bfcl_v3.py`** — Berkeley Function Calling Leaderboard v3 harness. Runs base, SFT, and SFT+DPO models across all four BFCL categories (simple, parallel, multiple, multi-turn). Produces a structured eval report.
- **`mmlu_regression.py`** — MMLU regression check. Guardrail to confirm SFT+DPO did not collapse general capability (target: within 2 points of base).
- **`compare.py`** — produces the model-vs-model comparison tables that go into the model card and technical report.
- **`results/`** — committed evaluation results (JSON reports + Markdown summaries), one file per model×benchmark×date.

## Reused by release-kit

The BFCL harness here is the first concrete input to release-kit's eval-harness pillar. Both projects call the same code; release-kit wraps it in the checklist-generation workflow.

## Not in v1

- HumanEval / code-generation benchmarks (out of scope — task is tool-calling)
- Long-context evals (not our failure surface)
- Custom LLM-as-judge (release-kit will contribute this; llama-tools consumes it if we need it)

## Related

- `../train/` — where checkpoints are produced
- `../docs/decisions/ADR-002-training-method-sft-dpo.md` — success criteria specified here
