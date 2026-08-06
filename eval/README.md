# eval/

Evaluation harness and evidence index used for the study-1 ship decision and study-2 preparation.

## Current components

- **`bfcl_simple.py`** — the study-1 BFCL v4 `simple_python` SFT/DPO sweep. It uses strict function-name and accepted-argument matching; it is not the official BFCL AST scorer.
- **`mmlu_regression.py`** — 5-shot MMLU capability-retention check.
- **`bfcl_leakage_check.py`** — exact and MinHash overlap checks between training prompts and cached BFCL questions.
- **`fetch_pinned_bfcl.py`** + **`manifests/bfcl_v4_study2.json`** — content-addressed acquisition and verification for the exact held-out inputs/keys used going forward.
- **`results/`** — small committed reports plus hashes and locations for larger public evidence.

Raw BFCL data stays under the gitignored `eval/bfcl_data/` cache. Training code must not consume it. Mining may use a frozen list of those files only for deterministic decontamination; evaluation uses them only as held-out inputs/keys.

## Reused by release-kit

The BFCL harness here is the first concrete input to release-kit's eval-harness pillar. Both projects call the same code; release-kit wraps it in the checklist-generation workflow.

## Not in v1

- HumanEval / code-generation benchmarks (out of scope — task is tool-calling)
- Long-context evals (not our failure surface)
- Custom LLM-as-judge (release-kit will contribute this; llama-tools consumes it if we need it)

## Related

- `../train/` — where checkpoints are produced
- `../docs/decisions/ADR-008-dpo-v2-negative-result.md` — accepted study-1 decision and result interpretation
