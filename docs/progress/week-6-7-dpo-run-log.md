# Week 6-7 run log — DPO v1 full run, checkpoint sweep, negative-result verdict

Continuation of [`week-4-run-log.md`](./week-4-run-log.md). The Week 5 smoke and pre-registered evaluation framing are recorded there (18:42/18:43 entries, 2026-07-19).

## 2026-07-19 — DPO v1 full run

- Full run launched (`train/dpo_full.py`): 9,942 train pairs / 300 holdout (seed 42), effective batch 16, ~622 steps planned, 1 epoch, lr 5e-6, beta 0.1, 1x RTX A6000 48GB.
- As pre-registered: `rewards/accuracies` pinned at 1.0 throughout — rule-based rejecteds mechanically easy for the SFT policy. Health signal tracked instead: `eval_rewards/chosen`.
- **Chosen-reward drift (the story of the run):** −0.36 (step 200) → −0.42 (step 300) → −0.53 (step 400), heading toward the −0.6 kill line. Margins near-identical from step 200 onward — no offsetting gain.
- Infra: pod throughput degraded ~3x mid-run (host contention, same failure mode as Week 4's 28→47 s/it episode).
- **~21:18 — EARLY STOP at step 400/622 (64%).** Rationale: converged (margins flat), chosen-reward degrading monotonically, remaining steps negative-expected-value at 3x cost. Checkpoints 100/200/300/400 verified uploaded to HF staging (`dpo-checkpoints/`, 13 files each). Pod stopped, billing off.

## 2026-07-20 — checkpoint sweep and verdict

- **07:41** — fresh A6000 pod up; repo cloned, deps installed, SFT adapter (9 files) + all 4 checkpoints (52 files, 3.56GB) pulled from HF; `preferences_dpo.jsonl` pulled from staging (byte-identical, 24,498,095 bytes).
- **07:51 — sweep complete** (`eval/dpo_sweep.py`, 5 candidates × 20 holdout prompts, greedy):

  | candidate | exact match | json valid |
  |---|---|---|
  | sft | 16/20 | 20/20 |
  | dpo-100 | 15/20 | 20/20 |
  | dpo-200 | 14/20 | 20/20 |
  | dpo-300 | 14/20 | 20/20 |
  | dpo-400 | 11/20 | 18/20 |

  Monotonic degradation with training steps — exactly what the chosen-reward drift predicted. dpo-400 emits invalid JSON (2/20): first hard evidence of real policy damage.
- **07:53 — human diff-read of every divergent generation** (report uploaded to HF staging `dpo-sweep/`). Verdict got cleaner, not murkier: SFT's own misses are largely formatting (argument order); DPO regressions are real (wiki-URL degraded to bare entity from checkpoint-100 on; exchange value corrupted `KRAKEN`→`BINANCE` from checkpoint-200 on; dpo-400 shows garbage tokens, stringified dicts, hallucinated schema fields). Prompts 7/8/10/19 fail for all candidates including SFT — hard prompts, not signal.
- **07:53 — VERDICT: SFT wins outright. No DPO checkpoint merges.** DPO v1 closed as a documented negative result → [ADR-006](../decisions/ADR-006-dpo-v1-negative-result.md). Shipped model remains `centuriandip/llama-3.1-8b-tools-sft` (Week 4 release, signed).
- **07:54 — pod stopped.** Morning spend ≈ $0.50. Nothing merged because nothing earned it.
- Root cause on record (ADR-006): mechanically-easy rule-perturbed pairs → margin already saturated → optimizer degrades the policy instead of ranking. The pre-registered health metric caught it mid-run; the sweep confirmed it; the diff-read widened SFT's lead.
- **Decision: DPO v2 approved as optional scope** → [ADR-007](../decisions/ADR-007-dpo-v2-on-policy-rejecteds.md). On-policy rejecteds sampled from the SFT model's own failures, tightened kill line (−0.25), pre-registered abort conditions at every stage, ≥1,500-pair floor. Pipeline code prepared today; GPU stages await next pod session.

## Week 6-7 arc, closed

What this arc demonstrates, in one paragraph: the preference data was audited (ADR-005, human adjudication), the run's success criteria were pre-registered *before* launch so we could not grade ourselves on an easy metric, the health metric caught policy degradation mid-run, the run was stopped early on evidence, a checkpoint sweep with human diff-read settled the verdict, and the honest answer — the cheaper, simpler SFT model is better — shipped. The negative result is the artifact.
