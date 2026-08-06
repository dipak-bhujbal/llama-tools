# Committed evaluation results

This directory contains small, reviewable result reports and a content-addressed index of larger evidence. Numbers are copied from run artifacts; planning targets and chat-only recollections do not belong here.

## Study 1

- [`study1_bfcl_simple_report.md`](./study1_bfcl_simple_report.md) is byte-identical to the report in the public evidence dataset at revision `a3905a7381fd8bcbb16a04081cad595da6c7e616`. Its 1,600 per-candidate generation rows remain in that dataset and are pinned by SHA-256 in [`evidence.json`](./evidence.json).
- [`study1_mmlu_smoke_report.md`](./study1_mmlu_smoke_report.md) is the 200-item smoke, not the full MMLU run. Its 1,000 per-candidate predictions are likewise pinned in the evidence index.
- The full 14,042-item MMLU values in ADR-008 do not currently have a recoverable raw report or prediction artifact in the repo or public evidence snapshot. The gap is explicit in `evidence.json`; the 200-item smoke must never be presented as the full run.

The held-out BFCL inputs and answer keys are not vendored. [`../manifests/bfcl_v4_study2.json`](../manifests/bfcl_v4_study2.json) and [`../fetch_pinned_bfcl.py`](../fetch_pinned_bfcl.py) reproduce and verify the exact files.
