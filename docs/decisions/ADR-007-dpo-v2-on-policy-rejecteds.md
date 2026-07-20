# ADR-007: DPO v2 — on-policy rejecteds sampled from the SFT model's own failures

**Status:** Accepted
**Date:** 2026-07-20
**Decision:** Build the v2 preference set by sampling the SFT policy at temperature on its own training prompts, keeping genuine failures (semantically wrong or unparseable generations) as rejecteds against the gold chosen. Train DPO v2 on these hard pairs only, with tightened pre-registered health metrics. Explicitly optional scope: if v2 also fails to beat SFT on the sweep, the project ships SFT (ADR-006) and stops.

## Context

ADR-006 closed DPO v1 as a negative result. Root cause: rule-perturbed rejecteds were trivially separable (`rewards/accuracies` pinned at 1.0, `logps/rejected` ≈ −40), so the optimizer had no ranking work to do and degraded the policy instead. The fix is not more training or different hyperparameters — it is preference data the model actually finds hard.

The hardest possible negatives for a policy are its own mistakes. A rejected sampled from the SFT model sits, by construction, in a high-probability region of the policy (`logps/rejected` comparable to `logps/chosen`), so ranking chosen above it requires a real update rather than a free margin.

This was pre-registered as the contingency in the Week 5 run-log (18:43 entry) and in ADR-004's revisit triggers.

## Options considered

### Option A: Retune v1 — lower beta, fewer steps, lower LR on the same data

Rejected. The sweep shows degradation already at checkpoint-100 (the wiki-URL regression). There is no step count at which this data helps; gentler optimization just degrades more slowly. The data is the problem.

### Option B: LLM-generated rejecteds (frontier model writes plausible-but-wrong calls)

Rejected for v2 (was also rejected for v1 in ADR-004). Cost and prompt-engineering surface are higher than sampling, the failure taxonomy is post-hoc rather than measured, and the negatives are still off-policy — plausible to the *generator*, not necessarily probable under *our* policy. On-policy sampling gives harder negatives for free and a taxonomy grounded in the model's real error distribution.

### Option C (chosen): On-policy rejecteds from temperature sampling

For each training prompt, sample K completions from the SFT policy (T=0.8, top-p 0.95). Score each against gold with semantic comparison (canonical JSON equality, not string match). Keep failures as rejecteds; pair with the gold chosen. Train DPO on the resulting hard pairs.

Known limitation, accepted: pair count is bounded by the model's failure rate — we get however many hard pairs exist, not a target count. That is the point. ADR-005's ≥10K floor applied to easy pairs; hard pairs carry far more signal per pair, and published DPO results regularly work at 2-5K pairs. A floor of **≥1,500 pairs** gates the training run (below that, the run is cancelled and the project ships SFT).

## Design

### Stage 1 — sample failures (`data/sample_failures.py`, GPU)

- Policy: base + `outputs/sft-full` adapter, identical to the sweep's SFT candidate.
- Prompts: the **training split only** of `preferences_dpo.jsonl` — same `train_test_split(test_size=300, seed=42)` as `train/dpo_full.py` and `eval/dpo_sweep.py`. The 300-prompt holdout is never sampled, so the final v1-vs-v2 sweep stays uncontaminated.
- Deduplicate prompts (v1 pairs reused source examples; we sample each unique prompt once).
- K=4 samples per prompt at T=0.8 / top-p 0.95, batched left-padded generation.
- Score each sample vs gold `chosen`:
  - `invalid_json` — output does not parse as a tool-call structure
  - `wrong_tool` — parsed, but tool name set differs from gold
  - `wrong_args` — right tools, wrong/missing/extra argument values
  - `ok` — canonically equal to gold (key order and whitespace ignored) → not a failure
- Write every non-`ok` sample as a candidate rejected with its failure category.

### Stage 2 — build pairs (`data/build_dpo_v2_pairs.py`, CPU)

- One pair per (prompt, distinct failure): chosen = gold, rejected = sampled failure.
- Dedupe canonically-identical rejecteds per prompt; cap at 2 pairs per prompt to bound per-prompt weight.
- Reuse ADR-005 triviality guards where they apply: drop pairs whose rejected is canonically equal to chosen (defensive; scoring should have caught it), keep unparseable rejecteds unconditionally (ADR-005's exemption logic, same rationale).
- Emit `data/processed/preferences_dpo_v2.jsonl` with schema `{prompt_messages, chosen, rejected, failure_type, source, source_id}` — drop-in compatible with the v1 trainer's loader.
- Emit a stats report: pairs per failure type, failure rate per prompt, dataset size vs the ≥1,500 floor.

### Stage 3 — train (`train/dpo_v2_full.py`, GPU)

Same pipeline as v1 with these deltas, each traceable to a v1 lesson:

| Setting | v1 | v2 | Why |
|---|---|---|---|
| Data | 10,242 easy pairs | hard pairs only | ADR-006 root cause |
| Kill line: `eval_rewards/chosen` | −0.6 | **−0.25** | v1 showed visible damage by −0.36; −0.6 was too permissive |
| Expected `rewards/accuracies` | pinned 1.0 (no signal) | **must start well below 1.0** — if it pins immediately, the pairs are still too easy → abort | direct test of the v2 hypothesis |
| `load_best_model_at_end` | unset (Week 4 lesson, repeated in v1) | set, `metric_for_best_model=eval_loss` | Week 4 run-log 15:43 |
| Eval/save cadence | 50/100 | 25/50 | smaller dataset → finer checkpoints for the sweep |
| Epochs / LR / beta | 1 / 5e-6 / 0.1 | unchanged | isolate the data variable — only change one thing |

### Stage 4 — judge (existing `eval/dpo_sweep.py`)

Same sweep, same 300-prompt holdout, candidates = SFT baseline + v2 checkpoints. Decision rule, pre-registered here: **v2 ships only if it beats SFT on the sweep after diff-read** — strictly more semantic matches, zero JSON validity regressions. Tie or loss → project ships SFT, DPO chapter closes for good with two documented negative results.

## Cost and schedule

- Stage 1: ~1-1.5 h GPU (≈8-9K unique prompts × 4 samples, batched, A6000) ≈ $0.75
- Stage 3: dependent on pair count; at ~2-4K pairs ≈ 1-2 h ≈ $1
- Stage 4: ~40 min ≈ $0.35
- Total ≈ $2-3, well inside budget. Stages 1-2 can run in one pod session with Stage 3 in the same session if the stats gate passes.

## Consequences

- `data/sample_failures.py` and `data/build_dpo_v2_pairs.py` are the canonical v2 data pipeline; `train/dpo_v2_full.py` the trainer.
- The v2 dataset's failure taxonomy (invalid_json / wrong_tool / wrong_args rates) is itself a publishable artifact: it measures the SFT model's true error distribution on its own training distribution.
- `train/merge_dpo_winner.py` serves the v2 winner if there is one.
- Abort conditions are pre-registered and mechanical: <1,500 pairs at Stage 2, accuracies pinned at Stage 3 start, chosen-reward < −0.25 mid-run, sweep tie/loss at Stage 4. Any of them → ship SFT, stop.

## Revisit trigger

- If Stage 1 yields too few failures because greedy-vs-sampled behavior diverges (model rarely fails even at T=0.8), revisit with higher temperature or adversarial prompt selection before abandoning.
- If v2 wins the sweep but BFCL (Week 7+) shows regression vs SFT, the sweep's 20-prompt scale is the suspect — re-judge on the full 300-prompt holdout before shipping.
