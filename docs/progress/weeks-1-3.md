# Progress report: Weeks 1-3

Covers 2026-07-18 (Weeks 1-3 executed in a single day against the [12-week plan](../../PLAN.md)).

## Summary

| Week | Focus | Status | Spend |
|---|---|---|---|
| 1 | Fundamentals + environment | Complete | ~$0 |
| 2 | SFT smoke test on cloud GPU | Complete | ~$0.17 |
| 3 | Data curation + preference pairs | Complete | $0 |

**Cumulative spend: ~$0.17 of the $1000 budget.**

## Week 1 — Fundamentals + environment

- Local Python 3.13 venv with transformers 5.14.1 / torch 2.13.0 / huggingface_hub 1.24.0
- HF auth configured (`centuriandip`), Llama-3.1-8B-Instruct license accepted and access verified via config-file download
- Smoke test: local CPU inference on `HuggingFaceTB/SmolLM2-135M-Instruct`

**Lesson — gated models:** the smoke test originally targeted `Llama-3.2-1B-Instruct`, but each Meta Llama family requires a separate HF license approval. Swapped to SmolLM2 (non-gated): the smoke test's job is to verify the environment, not gated-model access. Llama-3.1-8B (the actual training target) is verified separately.

**Lesson — transformers 5.x API:** `apply_chat_template(..., return_dict=True)` returns a dict; `device_map=` requires `accelerate`. Both handled in `smoke.py`.

## Week 2 — SFT smoke test

Run: LoRA-SFT of Llama-3.1-8B-Instruct on 500 examples of `NousResearch/hermes-function-calling-v1`, 1x RTX A6000 (48GB) on Runpod @ $0.49/hr.

| Metric | Value |
|---|---|
| Steps | 29 (450 train / 50 eval, effective batch 16) |
| Wall clock | 13m 22s (~27.7 s/step) |
| Train loss | 1.48 → 0.80 (best), 1.019 (epoch avg) |
| Eval loss | 0.897 (below train avg — no overfitting) |
| Eval token accuracy | 78.2% |
| Cost | ~$0.17 (incl. setup + upload) |

Config: LoRA rank 64 / alpha 128 targeting `q/k/v/o_proj`, lr 2e-4 cosine + 3% warmup, bf16, gradient checkpointing.

Artifact: `centuriandip/llama-3.1-8b-tools-week2-smoke` (private HF repo, adapter + optimizer state, 839MB).

**Lesson — version drift:** `trainer.push_to_hub()` failed with `create_model_card() got an unexpected keyword argument 'repo_id'` (TRL 1.8 vs bundled transformers). Fixed by switching to direct `HfApi.create_repo` + `upload_folder`, which is stable across ecosystem versions.

## Week 3 — Data curation

Pipeline: `assemble_sft.py` → `dedupe.py` → `assemble_preferences.py` → `push_to_hf.py`. All local, deterministic (seed 42), zero cloud spend.

### Assembly ([ADR-003](../decisions/ADR-003-source-datasets.md))

| Source | Loaded | Sampled |
|---|---|---|
| Hermes (`func_calling` + `func_calling_singleturn`) | 3,786 | 3,786 (all — under target) |
| xLAM (`Salesforce/xlam-function-calling-60k`) | 60,000 | 11,214 (backfilled to hit 15K total) |
| **Total** | | **15,000** |

**Lesson — dataset yields:** Hermes tool-use configs are ~1.9K each, far below initial estimates. The xLAM target is dynamic (`FINAL_TARGET - hermes_count`) so the total holds. xLAM is gated (Salesforce approval required — granted same-day); the script degrades gracefully to Hermes-only when access is missing.

### Dedup (MinHash+LSH, Jaccard 0.7, 128 perms, word-level 5-grams)

| | Input | Kept | Dropped |
|---|---|---|---|
| Total | 15,000 | 12,160 | 2,840 |
| Hermes | 3,786 | 1,161 | 2,625 (69%) |
| xLAM | 11,214 | 10,999 | 215 (1.9%) |

**Analysis:** the Hermes drop rate reflects genuinely templated queries in `func_calling_singleturn`, not a dedup bug. Post-dedup composition is ~90% xLAM / 10% Hermes — a deviation from ADR-003's diversity goal, accepted deliberately (unique-query signal > headcount). Revisit dedup threshold only if Week 4 training quality is weak.

**Deferral:** BFCL v3 leakage dedup deferred to Week 7 (concurrent with the eval harness) — rationale documented in `data/dedupe.py`.

### Preference pairs ([ADR-004](../decisions/ADR-004-preference-synthesis.md))

10,999 pairs (1:1 chosen:rejected) from the xLAM subset via rule-based adversarial perturbation:

| Perturbation | Count | Share |
|---|---|---|
| malformed_json | 2,401 | 21.8% |
| wrong_arg_value | 2,361 | 21.5% |
| missing_required_arg | 2,329 | 21.2% |
| hallucinated_tool | 2,318 | 21.1% |
| wrong_tool_from_list | 1,590 | 14.5% |

**Lessons:** (1) multi-call examples (list-shaped answers, 54% of xLAM) initially skipped — now the first call is perturbed and the rest preserved; (2) `missing_required_arg` fired 0% until a fallback (drop any provided arg) was added, since many xLAM schemas don't declare `required`.

Artifact: [`centuriandip/tool-calling-preferences`](https://huggingface.co/datasets/centuriandip/tool-calling-preferences) (private until release) with a full dataset card: methodology, perturbation taxonomy, CC-BY-4.0 with xLAM attribution.

## Artifacts as of end of Week 3

1. `github.com/dipak-bhujbal/llama-tools` — public, working pipeline, 4 accepted ADRs
2. `github.com/dipak-bhujbal/release-kit` — private, 3 accepted ADRs, package scaffolding
3. `centuriandip/llama-3.1-8b-tools-week2-smoke` — private HF model (smoke adapter)
4. `centuriandip/tool-calling-preferences` — private HF dataset (10,999 DPO pairs)

## Next: Week 4

Full SFT run — 12,160 examples, 3 epochs, ~6-8 hrs on 1x GPU, estimated $60-100. Produces `centuriandip/llama-3.1-8b-tools-sft`, the first real model artifact.
