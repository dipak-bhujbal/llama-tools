# ADR-006: DPO v1 negative result — ship SFT as final, close DPO-on-rule-perturbed-preferences

**Status:** Accepted
**Date:** 2026-07-20
**Decision:** Do not merge any DPO v1 checkpoint. The Week 4 SFT model (`centuriandip/llama-3.1-8b-tools-sft`) remains the shipped model. DPO v1 is closed as a documented negative result: DPO on rule-perturbed preference pairs monotonically degraded the policy. A v2 with on-policy rejecteds is designed separately (ADR-007).

## Context

Week 6 ran full DPO (`train/dpo_full.py`) on the 10,242-pair preference set from ADR-004/ADR-005: chosen = gold tool call, rejected = one of five deterministic rule-based perturbations. The run was deliberately stopped at step 400/622 (64%) after convergence, with `eval_rewards/chosen` drifting negative: −0.36 (step 200) → −0.42 (step 300) → −0.53 (step 400), toward the pre-registered −0.6 kill line. Checkpoints 100/200/300/400 were vaulted to HF.

Training metrics could not pick a winner — margins were near-identical from step 200 onward while chosen-reward degraded — so Week 7 ran a generation sweep (`eval/dpo_sweep.py`): greedy completions from the SFT baseline plus all four checkpoints on 20 prompts from the trainer's own 300-pair holdout (identical seed/split).

### Sweep results (2026-07-20)

| candidate | exact match | json valid |
|---|---|---|
| sft | 16/20 | 20/20 |
| dpo-100 | 15/20 | 20/20 |
| dpo-200 | 14/20 | 20/20 |
| dpo-300 | 14/20 | 20/20 |
| dpo-400 | 11/20 | 18/20 |

Degradation is monotonic in training steps. A human diff-read of every divergent generation (report at HF staging `dpo-sweep/sweep_report.md`) confirmed the misses are real regressions, not formatting noise:

- **Prompt 2:** SFT emits the correct full wiki URL; all four DPO checkpoints degrade it to a bare entity name — regression present already at checkpoint-100.
- **Prompt 6:** SFT's "miss" is argument order only (semantically identical); dpo-200/300/400 change the exchange value (`KRAKEN:ETHUSD` → `BINANCE:ETHUSD`) — factually wrong.
- **Prompt 19:** dpo-200/400 null out a populated `content_type` field — information loss.
- **checkpoint-400 is corrupted outright:** garbage token sequences (`"name": "x",","`), dict arguments cast to strings, hallucinated schema fields, coordinates rounded to integers, and 2/20 unparseable outputs. This is the −0.53 chosen-reward made visible.

Four prompts (7/8/10/19) fail for every candidate including SFT — hard prompts, not DPO signal.

## Options considered

### Option A: Merge the least-degraded checkpoint (dpo-100)

Rejected. dpo-100 is strictly worse than SFT (15/20 vs 16/20) and already carries the wiki-URL regression. Merging it would ship a measured regression to claim "SFT+DPO" on the label. The label is not worth the regression.

### Option B: Resume training to step 622 and re-evaluate

Rejected. Every signal — chosen-reward trajectory, monotonic sweep degradation, checkpoint-400's corruption — says more steps make it worse. Spending compute to confirm a trend that three independent measurements already agree on is not diligence, it is procrastination.

### Option C (chosen): Ship SFT as final; close DPO v1 as a negative result; design v2 separately

The SFT model wins outright on the sweep. It is already merged, signed, and released (Week 4, release-kit checklist sha256 45fbc788). No further compute is needed. The negative result is documented (this ADR), and the path to a DPO that could actually help — on-policy rejecteds — is specified in ADR-007 as an explicitly optional follow-up.

## Decision

Option C. The shipped model is and remains `centuriandip/llama-3.1-8b-tools-sft`.

## Root cause

The failure was predicted by the data, pre-registered before the run, and confirmed by three independent measurements:

1. **The rejecteds were mechanically easy.** Rule-based perturbations (ADR-004) are trivially separable for an SFT-ed policy: the Week 5 smoke showed `logps/rejected` ≈ −40 vs `logps/chosen` ≈ −0.4, and `rewards/accuracies` pinned at 1.0 from the first steps. There was almost no gradient signal available from ranking chosen above rejected — the model already ranked them correctly.
2. **With no ranking work left, the optimizer found the degenerate direction.** DPO maximizes the margin between chosen and rejected log-probabilities. When the margin is already saturated, continued optimization pushes *both* down with rejected falling faster — visible as `eval_rewards/chosen` drifting negative. The update was eating the SFT signal rather than adding preference signal on top of it.
3. **The pre-registered framing caught it.** The Week 5 run-log entry (18:43, before launch) declared pinned accuracies to be pipeline-health-only, located genuine signal in downstream generation quality, and named the contingency: "v2 preference set with model-generated rejecteds (harder negatives)." The sweep executed that evaluation and the contingency is now triggered.

The known limitation recorded in ADR-004 ("rule-based rejecteds are deterministic and documentable" traded against realism) was the right call for v1 — it made the dataset auditable and the failure analyzable — but the realism cost turned out to dominate the training outcome.

## Consequences

- `centuriandip/llama-3.1-8b-tools` (the planned SFT+DPO repo) stays unpublished. If DPO v2 succeeds it ships there; otherwise the SFT model is the project's final model.
- The four DPO v1 checkpoints stay vaulted in HF staging (`dpo-checkpoints/`) as evidence, not candidates.
- Sweep artifacts (`generations.jsonl`, `sweep_report.md`) are archived at HF staging `dpo-sweep/`.
- `train/merge_dpo_winner.py` is not run for v1. It remains valid for a future v2 winner.
- README and model card must present DPO v1 as a negative result, not omit it. The eval story ("pre-registered health metric → early stop → checkpoint sweep → honest verdict") is a primary artifact of this project.
- Week 7 BFCL evaluation proceeds with two candidates (base, SFT) instead of three.
- ADR-007 specifies the v2 design. It is optional scope: SFT already clears the project's credential bar.

## Revisit trigger

Revisit only if the sweep methodology is shown to be flawed — e.g., the holdout split in `eval/dpo_sweep.py` is found not to match the trainer's split (both derive from `train_test_split(test_size=300, seed=42)`; verified identical at sweep time), or exact-match scoring is shown to have mis-graded the diff-read prompts. Neither is expected: the diff-read was a manual semantic re-scoring precisely to remove metric artifacts, and it widened SFT's lead rather than narrowing it.
