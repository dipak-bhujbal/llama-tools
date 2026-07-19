# ADR-005: Preference-pair triviality filtering — exempt unparseable-rejected pairs from edit-distance filter

**Status:** Accepted
**Date:** 2026-07-19
**Decision:** Filter trivially-different preference pairs before DPO using exact-match, canonicalized-JSON equality, and a near-identical edit-distance ratio (≥0.98), but exempt pairs whose rejected side fails to parse as JSON (while the chosen side parses) from the edit-distance filter only.

## Context

Week 3 produced 10,999 preference pairs (chosen = correct tool call, rejected = one of five rule-based perturbations from ADR-004). DPO training (Week 6) learns from the *difference* between chosen and rejected. If that difference is negligible — identical text, whitespace-only key reordering, or a one-character edit that still produces a semantically equivalent parsed value — the model receives zero gradient signal and the pair dilutes the dataset.

Week 5 task: filter trivial pairs before the DPO run while keeping ≥10K final pairs (the minimum for a meaningful DPO signal at this model scale).

Three signals flag a pair as trivial (implemented in `data/validate_preferences.py`):

1. **exact_match** — raw chosen and rejected strings are byte-for-byte identical.
2. **canonical_json_equal** — both sides parse as JSON and, after recursive key-sort, yield equal structures (catches whitespace and key-ordering differences).
3. **near_identical_edit_distance** — `SequenceMatcher` ratio ≥ 0.98 (catches one- or two-character edits that still leave the text nearly the same).

A pair is filtered if it triggers any signal. The question is which signals apply to which pairs.

## Options considered

### Option A: Strict filter — apply all three signals uniformly, then synthesize replacement pairs

Apply exact_match, canonical_json_equal, and near_identical_edit_distance to every pair with no exemptions. Where the resulting count falls below 10K, run `assemble_preferences.py` again to synthesize additional pairs until the target is met.

Rejected. Strict filtering removes 2,394 of 2,401 `malformed_json` pairs (99.7%) because malformed-JSON perturbations are one-character edits by design (drop a brace, add a trailing comma, unterminate a string) — the edit-distance ratio is very high. Synthesizing more pairs to compensate papers over a measurement error rather than fixing it: the edit-distance ratio is measuring textual difference, but DPO needs semantic difference, and these pairs have maximal semantic difference (valid JSON vs. unparseable). Additional synthesis cost is also non-trivial and the replacement pairs would not cover the same failure mode.

### Option B: No triviality filtering at all

Keep all 10,999 pairs and let DPO train on everything.

Rejected. Pairs that are truly identical or differ only in key order or whitespace give DPO zero gradient signal. Including them pads the batch count without adding training signal and dilutes the loss relative to genuinely contrastive pairs. The exact_match and canonical_json_equal signals catch real data-quality problems (perturbation failed silently, or a value swap happened to produce the same canonical structure) that should not reach DPO training.

### Option C (chosen): Selective exemption — exempt pairs whose rejected side fails to parse from the edit-distance filter only

Apply all three signals normally, but suppress the near_identical_edit_distance reason for any pair where the chosen side parses as valid JSON and the rejected side does not. The exact_match and canonical_json_equal signals still apply to those pairs unconditionally.

The exemption logic in `triviality_signals()`:

```python
exempt_unparseable = c_canon is not None and r_canon is None

if near_identical and not exact and not exempt_unparseable:
    trivial_reasons.append("near_identical_edit_distance")
```

`_canonicalize()` returns `None` on `json.JSONDecodeError`, so `r_canon is None` is the exact condition for "rejected side is malformed JSON."

## Decision

Option C: selective exemption for unparseable-rejected pairs from the edit-distance filter.

## Rationale

### Why edit distance is the wrong measure for malformed-JSON pairs

Edit distance measures textual difference. A one-character brace deletion (`{"name": "foo", "arguments": {"x": 1}` missing the closing `}`) produces a ratio very close to 1.0 — textually near-identical. But the semantic difference is maximal: one side is a valid, executable tool call; the other cannot be parsed at all and represents a total model failure. This is precisely the failure mode the `malformed_json` perturbation class exists to teach the model to avoid.

Applying the edit-distance filter to these pairs confuses "the strings look similar" with "the model would learn nothing." The model absolutely can learn something: "output that differs from the correct call by a single missing character is categorically wrong, not just slightly wrong."

### Why the exemption is narrow

The exemption applies only to the edit-distance signal, and only when two conditions both hold: (a) the chosen side parses and (b) the rejected side does not parse. It does not exempt pairs from exact_match or canonical_json_equal. It does not exempt pairs where both sides parse — a `wrong_arg_value` pair with "5" vs "5.0" is semantically trivial and the edit-distance filter correctly removes it (705 pairs removed from `wrong_arg_value`; kept intentionally).

This narrowness means the exemption cannot accidentally preserve pairs that are genuinely uninformative.

### Counts

| | Input | Filtered | Kept |
|---|---|---|---|
| Total | 10,999 | 742 | 10,257 |

Filtered by reason (a pair can match multiple reasons):

| Reason | Count |
|---|---|
| near_identical_edit_distance | 683 |
| exact_match | 59 |
| canonical_json_equal | 59 |
| exempt_unparseable_rejected (spared from filter) | 2,394 |

Per-perturbation breakdown (notable cases):

- `malformed_json`: 2,394 pairs spared by exemption; small residual filtered by exact_match or canonical_json_equal.
- `wrong_arg_value`: 705 pairs removed by near_identical_edit_distance — correct behavior, a parsed value swap of "5" vs "5.0" is semantically trivial.
- Final count 10,257 clears the ≥10K target without additional synthesis.

### Human validation

Running `python data/validate_preferences.py --sample 200` produces a 200-pair stratified spot-check sample: 40 pairs per perturbation type, fixed seed (42), drawn from the full pre-filter set. The sample is rendered to `data/processed/spot_check_sample.md` with fields for manual `Verdict:` annotation. Sampling from the pre-filter set (not just kept pairs) allows reviewers to catch trivial pairs that the automated filter may have missed, not only pairs the filter correctly removed.

## Consequences

- `data/validate_preferences.py` is the canonical implementation. The `triviality_signals()` function is the single definition of "trivial" for this project.
- `data/processed/preferences_dpo.jsonl` (10,257 pairs) is the DPO training input for Week 6.
- The `perturbation_type` field is preserved on all kept pairs so Week 6 reward-accuracy analysis can break down DPO learning per failure mode.
- Model card must document the filtering methodology and the exemption rationale, particularly the semantic-vs-textual-distance distinction for malformed-JSON pairs.
- If Week 6 DPO metrics show poor reward accuracy on `malformed_json` specifically, the first diagnostic is to verify the exempted pairs are reaching the trainer as expected (not being silently dropped by a downstream step).

## Revisit trigger

Revisit if:

- Week 6 DPO reward accuracy on `malformed_json` is anomalously low compared to other perturbation types — may indicate the exemption logic has a bug or the trainer is handling unparseable rejected samples unexpectedly.
- A future perturbation type produces one-character edits that are not malformed-JSON (e.g., a single-digit value change that happens to also be unparseable for a different reason) — the exemption condition should be checked for false positives.
- The target pair count floor is raised above 10,257, requiring a review of whether further filtering is feasible.
