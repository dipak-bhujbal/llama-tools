# Quarantined chat attachments — 2026-08-04

These files were copied, without modification, from
`/Users/dipak.bhujbal/Downloads/files` after the owner explicitly directed
their intake in agentchattr message 2006. They remain untrusted intake and are
not training data or measured evidence.

## Receipt

| File | Rows / lines | SHA-256 |
|---|---:|---|
| `scale_pairs_fixed.py` | 1,050 lines | `53f3b7b9b1df420781bf1274cf45231bd35e935447e6d1381c5a8fbf39052810` |
| `fixture_pairs_train.jsonl` | 1,440 rows | `6b0215aae4bf18c806954916bbd889df0d8de1019292af2169cc86f35f0b8d04` |
| `fixture_pairs_eval.jsonl` | 160 rows | `ae862c34c216c01f99a263400684fcbb1229e8ece7cf17e8c297c050c8232807` |
| `fixture_audit_sample_50.jsonl` | 50 rows | `88e56ef65f18a7be7506de08f89d219ff9075e7f118cc9f27a774f764665265f` |

The expected fifth attachment, `mine_pairs.py`, was absent from the supplied
folder. An exact-name and wildcard search under `~/Downloads` found no copy.

## Read-only intake checks

- All three JSONL files parse one JSON object per line.
- Pair IDs and prompts are unique within each file.
- Train and evaluation IDs are disjoint; every audit ID is in train.
- Every row is marked `meta.synthetic: true` and contains no `pass_rate`,
  `verified_by`, `gen_temperature`, or `source_dataset` metadata field.
- Combined train/evaluation error-type counts match the stated 1,600-row
  allocation.
- The Python source parses as valid Python and its imports are standard-library
  only. It was not executed during intake.

## Issues to resolve before promotion

- `mine_pairs.py` must be supplied separately.
- The generator writes to absolute `/home/claude/dpo_review/...` paths instead
  of repository-relative or caller-supplied paths.
- Fixture provenance names `scale_pairs.py`, not the supplied
  `scale_pairs_fixed.py`.
- The delivered fixture rows must be regenerated and compared byte-for-byte
  after the generator is adapted and reviewed; supplied claims do not become
  repository evidence merely because the files were received.
