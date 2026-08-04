# Study 2 Phase 0 — verified study-1 record

**Audit date:** 2026-08-04  
**Starting repository commit:** `ef41eef`  
**Method:** reconstruct from committed scripts, accepted ADRs, run logs, and content-addressed evaluation artifacts. Conversation memory is not evidence.

## Corrected record

| Item | Verified record | Primary in-repo source |
|---|---|---|
| Week-2 SFT smoke | 500 examples split 450 train / 50 eval; one epoch; 29 optimizer steps; effective batch 16 | `train/sft_smoke.py`, `docs/progress/weeks-1-3.md` |
| Week-4 full SFT | 12,160 examples split 11,660 / 500; three epochs; 1,095 steps; effective batch 32; final eval loss 0.21165 | `train/sft_full.py`, `docs/progress/week-4-run-log.md` |
| DPO v1 | 10,242 pairs split 9,942 / 300; effective batch 16; 622 steps planned; stopped at 400; eval cadence 50 | `train/dpo_full.py`, `docs/progress/week-6-7-dpo-run-log.md`, ADR-006 |
| DPO v2 | 2,523 pairs; effective batch 16; 150 steps; eval cadence 25 | `train/dpo_v2_full.py`, ADR-008 |
| BFCL v4 simple_python | SFT 369/400 = 92.25%; DPO checkpoints 364/400, 363/400, 359/400 | `eval/results/study1_bfcl_simple_report.md` plus pinned public generations |
| MMLU full | base 0.683; SFT 0.659; DPO-50 0.658, as recorded in the accepted ADR | ADR-008; raw full-run artifact not recovered (see `eval/results/evidence.json`) |

## BFCL baseline correction and freeze

The study-1 question file contains 400 JSON objects and 400 unique ids. The old `n=399` text came from counting newline characters in a file without a trailing newline; the loader strips blank lines and scores all 400 objects.

The original BFCL v4 release commit (`58f57e9`) is not the exact study-1 data revision: its `simple_python_363` answer names `find_closest`, while the study-1 key names `restaurant_search.find_closest`. Upstream data-fix commit `9d8416a` matches both study-1 files byte-for-byte. The per-file blob ids, SHA-256s, row counts, unique-id counts, and id-set digests in `eval/manifests/bfcl_v4_study2.json` are the binding pins; commit ids are provenance annotations.

## Evidence boundary

The BFCL report is backed by 1,600 public per-item generation rows. The public MMLU artifact is only the 200-item smoke. ADR-008 records the full 14,042-item MMLU results, but its raw report and prediction rows were not recovered during this audit. No reconstructed file may be presented as the missing run artifact.

## Fabricated-meta purge inventory

At the start of Phase 0, none of these legacy names existed in the working tree or any Git ref: `scale_pairs.py`, `dpo_pairs_train.jsonl`, `dpo_pairs_eval.jsonl`, `audit_sample_50.jsonl`, `DPO_pairs_data.zip`. No JSONL contained `exact_match_checker_v2`. The purge is therefore a verified no-op for the current repository, not a claim about an external sandbox or future supplied bundle.

`scripts/check_artifact_boundaries.py` enforces the current-tree boundary. Any later external bundle is untrusted intake and must be inventoried before selected files are copied into the branch.
