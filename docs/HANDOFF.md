# HANDOFF.md - llama-tools rescue study (DPO round 2)

**Repo:** `github.com/dipak-bhujbal/llama-tools`
**Owner:** Dipak (human). Agents execute; the human signs off at every DECISION GATE.
**Last updated:** 2026-08-04

---

## 0. Mission in one paragraph

Study 1 shipped an SFT-tuned Llama-3.1-8B tool-calling model (92.25% BFCL
simple_python on 400 held-out examples; MMLU 0.659 vs 0.683 base) and killed
two DPO variants against pre-registered abort criteria: training-time
preference margins improved while held-out tool-use came in below SFT at every
evaluated checkpoint.
The published ADR concluded DPO's preference signal is redundant when SFT is
at ceiling on ground-truth tasks. Study 2 (this handoff) tests whether DPO
becomes non-redundant under corrected conditions: on-policy hard negatives
mined from the model's own errors, a bounded/anchored objective, and
checkpoint selection driven by ground-truth evals instead of preference
metrics. The outcome of study 2 is NOT predetermined. Both "DPO now helps"
and "DPO still doesn't help" are acceptable, publishable endpoints.

---

## 1. GROUND RULES (non-negotiable, read before any task)

1. **No fabricated provenance, ever.** Every metadata field in every dataset
   row must describe something that actually happened (a real generation, a
   real verifier verdict, a real pass rate). The legacy synthetic set
   violated this and is being purged (Phase 0). Synthetic fixtures are fine
   ONLY when labeled `"synthetic": true` with honest provenance strings.
2. **Fixtures never train anything.** `tests/fixtures/fixture_pairs_*.jsonl`
   exist to unit-test the verifier and dry-run trainer configs. They must
   never appear in a training data path, an HF upload, or an evidence
   artifact.
3. **Results are outputs, not targets.** The figures floating around from
   planning docs (51->71 BFCL, KTO 69.5, SFT-only 67, beta 0.1 optimal) are
   UNVERIFIED HYPOTHESES. They must never be written into README, ADRs,
   resume drafts, HF cards, or commit messages as results. Only numbers with
   a run artifact behind them (ledger, trainer_state.json, eval JSON) may be
   reported. The measured public record today is: SFT 92.25% / MMLU -2.4 pts
   / DPO killed.
4. **Eval sets are read-only.** Nothing may edit, regenerate, or filter the
   held-out BFCL set (400 examples) or any eval split after it is frozen.
   Decontamination removes items from TRAINING pools only.
5. **Ledgers are the source of truth.** `mining_out/ledger.jsonl` and
   `trainer_state.json` files are append-only evidence. Never hand-edit.
6. **Human sign-off at every DECISION GATE** (marked [G] below). Agents stop
   and report; they do not proceed past a gate on their own.
7. **No pushes to HF or public GitHub without human approval.** Local commits
   on a branch are fine; publishing is a human action.
8. **Pre-registration before execution.** Any threshold, sweep value, or kill
   line used in Phases 3-5 must be written into `docs/prereg-study2.md` and
   committed BEFORE the corresponding run starts. If it is not in that file
   before the run, it cannot be used to select results after.

---

## 2. Current state

### 2.1 Measured record (study 1, closed)

| Item | Value | Evidence |
|---|---|---|
| SFT recipe | Llama-3.1-8B-Instruct + LoRA r64/alpha128 on q/k/v/o_proj, dropout 0.05, bf16, cosine LR + 3% warmup, TRL SFTTrainer | repo configs, wandb |
| SFT smoke | ~450 train / 50 eval, ran 3 epochs (config said 1; caught via trainer_state), no overfit: eval_loss ~0.212 vs train ~0.166 | trainer_state.json |
| Shipped SFT | **92.25% BFCL simple_python** (369/400 held-out); MMLU 0.659 vs base 0.683 (-2.4 pts, inside pre-set band) | eval artifacts |
| DPO v1 smoke | 32 steps, batch 2, no eval set; rewards/accuracies hit 1.0 almost immediately (easy-negatives flag) | outputs/dpo-smoke |
| DPO v2 | 622 steps, batch 2, peak LR 5e-06, ~1.2-1.3k pairs, eval every ~50 steps; margins grew past 9, logps/rejected drifted -28 to -135 unbounded | /workspace/keep/checkpoint-100..400+, logs |
| Kill decision | Both DPO variants closed as documented negative results: held-out tool-use came in below SFT at every evaluated checkpoint while margins improved | ADR |
| Published | `centuriandip/llama-3.1-8b-tools-dpo-v2-evidence` (HF), repo on GitHub | public |

### 2.2 New assets produced 2026-08-04

| File | Status | Purpose |
|---|---|---|
| `scale_pairs_fixed.py` | ready | Honest fixture generator: emits `"synthetic": true` + provenance string; fabricated fields removed |
| `fixture_pairs_train.jsonl` (1,440) | ready | Verifier unit tests + trainer dry-runs ONLY |
| `fixture_pairs_eval.jsonl` (160) | ready | same |
| `fixture_audit_sample_50.jsonl` (50) | ready | quick-test subset |
| `mine_pairs.py` | ready, validated | On-policy mining: verifier (onpolicy_verifier_v1), fixture self-test gate (1,600/1,600 pass, 0 false positives, 0 misses), Hermes loader, batched sampling, guardrailed pair building, ledger-based resume, `--redo-last N` rollback, `--fresh`, pre-registered decision output |

### 2.3 Known contamination to purge (Phase 0 blocker)

The ORIGINAL `scale_pairs.py` and its outputs (`dpo_pairs_train.jsonl`,
`dpo_pairs_eval.jsonl`, `audit_sample_50.jsonl`, any `DPO_pairs_data.zip`)
carry fabricated meta fields (`pass_rate` drawn from RNG, `verified_by`
naming a checker that never ran, `gen_temperature`, fake `source_dataset`).
These files must not survive anywhere reachable: workspace, repo going
forward, HF staging, local zips.

### 2.4 Required inputs (human must supply; agents must not guess)

- `SFT_ADAPTER`: path or HF id of the SHIPPED SFT adapter (the 92.25%
  checkpoint). Confirm with the human before the first mining run.
- `BFCL_LOCAL`: local path(s) to BFCL jsonl files for decontamination and
  the callback eval subset.
- HF auth token on the pod (gated Llama license already accepted).
- wandb auth if run tracking is wanted for study 2 (recommended).

---

## 3. ROADMAP

Legend: [A] agent-executable end to end, [H] human required,
[G] DECISION GATE (stop, report, wait for sign-off).

---

### PHASE 0 - Repo hygiene & purge (do first, ~30 min, $0)

- [A] 0.1 Create branch `study2/on-policy-dpo`.
- [A] 0.2 Delete legacy fabricated-meta files everywhere in the working
  tree: original `scale_pairs.py`, `dpo_pairs_train.jsonl`,
  `dpo_pairs_eval.jsonl`, `audit_sample_50.jsonl`, `DPO_pairs_data.zip`.
- [A] 0.3 Add new layout:
```
  mining/mine_pairs.py
  tests/fixtures/scale_pairs_fixed.py
  tests/fixtures/fixture_pairs_train.jsonl
  tests/fixtures/fixture_pairs_eval.jsonl
  tests/fixtures/fixture_audit_sample_50.jsonl
  docs/prereg-study2.md
  docs/HANDOFF.md   (this file)
```
- [A] 0.4 Add `tests/README.md` one-liner: fixtures are synthetic, for
  verifier tests and dry runs only, never training data.
- [A] 0.5 Guard script (pre-commit or CI): fail if any file under `data/`
  or any training config references `tests/fixtures/`.
- [A] 0.6 Commit: `study2: purge fabricated-meta synthetic set; add honest
  fixtures + mining pilot`.
- [H] 0.7 If the fabricated files were ever pushed to remote or HF: human
  decides whether history rewrite / HF deletion is needed. Agents flag it,
  never act on it.
- Acceptance: `grep -r "exact_match_checker_v2" --include="*.jsonl" .`
  returns nothing; self-test passes from new paths:
  `python mining/mine_pairs.py --self-test --fixtures tests/fixtures`.

---

### PHASE 1 - Mining pilot (~15 min pilot + ~1.5-3 h full; ~$1-4)

- [H] 1.1 Human confirms `SFT_ADAPTER` and provides `BFCL_LOCAL`.
- [A] 1.2 On the pod, inside tmux:
```
  python mining/mine_pairs.py --self-test --fixtures tests/fixtures
  python mining/mine_pairs.py --adapter $SFT_ADAPTER \
      --fixtures tests/fixtures --bfcl-path $BFCL_LOCAL \
      --n-prompts 100 --out-dir mining_pilot
```
- [A] 1.3 Report `mining_pilot/mining_summary.json` (histogram, yield rate,
  projections) to the human. Do not proceed on projections alone.
- [G] 1.4 GATE: human reads the pilot histogram, approves the full run.
- [A] 1.5 Full run (same command, `--n-prompts 1000 --out-dir mining_out`).
  Resume on interruption = rerun the same command (ledger-based). Rollback
  a suspicious item: `--redo-last N`.
- [A] 1.6 Commit `mining_out/ledger.jsonl` + `mining_summary.json` as
  evidence. Do not commit raw generations if they bloat the repo.
- Agent notes: never edit the ledger; never re-seed to "improve" yield; if
  the Hermes config name fails to load, try without a config and record
  which config was actually used in the commit message.

---

### PHASE 2 - DECISION GATE on yield (pre-registered, already fixed)

| Projected pairs @10k prompts | Decision |
|---|---|
| >= 1,000 | PROCEED to Phase 3A (DPO rerun) |
| 300-999 | PROCEED CAUTIOUSLY: 1 epoch max, eval callback every ~50 steps |
| < 300 | DO NOT run DPO. Go to Phase 3B (rejection-sampling SFT). This outcome is itself a publishable finding that STRENGTHENS the study-1 ADR |

- [A] 2.1 Apply the table mechanically from `mining_summary.json`.
- [H] 2.2 Human countersigns the branch taken (one dated line in
  `docs/prereg-study2.md`).
- [A] 2.3 Audit prep regardless of branch:
  `shuf -n 50 mining_out/mined_pairs.jsonl > mining_out/audit_read_me.jsonl`
  and surface the `ambiguous_review.jsonl` row count.
- [H] 2.4 Human reads the 50-pair audit + all ambiguous rows. Checks:
  (a) every "chosen" is actually correct (>3-5% verifier error poisons the
  signal), (b) rejected rows are genuinely wrong, not
  acceptable-but-different. Any verifier change requires re-running the
  fixture self-test AND re-mining affected prompts.

---

### PHASE 3A - DPO rerun track (only if Phase 2 says proceed)

Pre-registration first, runs second.

- [A] 3A.1 Write `docs/prereg-study2.md` BEFORE any training, containing:
  - Data: `mining_out/mined_pairs.jsonl` (git hash of ledger), 90/10
    train/eval split stratified by error_type, split seed fixed.
  - Arms (human may amend BEFORE first run, never after):
    beta in {0.05, 0.1, 0.3} plain DPO; one anchored arm (DPOP or
    DPO + SFT-loss mix on chosen, pick one and state it); optional IPO arm
    at one beta.
  - Fixed across arms: LoRA config identical to study 1, 1 epoch, batch 2
    (+grad accum as VRAM allows), peak LR 5e-06 unless re-justified in the
    doc, fixed seed.
  - Callback eval: a BFCL subset (~100 items) drawn ONLY from a frozen dev
    slice or categories NOT in the held-out 400 (document which), every
    ~50 steps.
  - KILL LINES per arm: (1) callback BFCL below SFT baseline -1.0 pt on two
    consecutive evals -> stop arm, mark killed; (2) eval logps/chosen falls
    more than a stated threshold below its step-0 value (state the number
    before running); (3) any NaN/inf.
  - Checkpoint selection: best callback-BFCL checkpoint per arm. NOT best
    eval_loss. NOT best margins.
- [H] 3A.2 Human approves the prereg doc. Commit. Runs may now start.
- [A] 3A.3 Implement `training/train_dpo_v3.py` (TRL DPOTrainer): eval
  split wired, callback eval as a TrainerCallback, kill lines as automatic
  early-stop writing `kill_report.json`, per-arm out-dirs
  `outputs/dpo-v3-<arm>/`, save_steps aligned with callback cadence, wandb
  tags per arm.
- [A] 3A.4 Dry-run the trainer on `tests/fixtures` for 5 steps to validate
  plumbing only (fixtures may exercise CODE, never produce a kept
  checkpoint; delete the dry-run output dir afterward).
- [A] 3A.5 Execute arms sequentially in tmux. After each arm: commit
  trainer_state.json + kill_report.json (if any) + the callback curve.
- [G] 3A.6 GATE: human reviews per-arm callback curves before Phase 5.
- Budget: ~1-3.5 h per arm on the current tier, ~$1-3/arm; full sweep
  (4-5 arms) ~$5-15.

---

### PHASE 3B - Rejection-sampling SFT track (if Phase 2 says no-DPO, or as
a parallel ablation arm later)

- [A] 3B.1 Build `data/rsft_train.jsonl` from `mining_out/sft_bucket.jsonl`
  (0/8 prompts + ground-truth answers), optionally plus verified-correct
  self-generations from the 1-7 zone (state the composition in prereg).
- [A] 3B.2 Continue SFT from the shipped adapter: identical LoRA config,
  1 epoch, LR 1e-4 to 2e-4 (state the chosen value in prereg), held-out
  eval split, BFCL callback identical to 3A.
- [A] 3B.3 Same kill lines, same checkpoint-selection rule.
- Budget: single run, likely under 1 h, ~$1.

---

### PHASE 4 - KTO comparison arm (optional, after 3A)

- [A] 4.1 Convert `mined_pairs.jsonl` to KTO's binary format (chosen ->
  desirable, rejected -> undesirable; keep pair_id linkage in meta).
- [A] 4.2 One KTO run, hyperparameters stated in prereg BEFORE running,
  same callback + kill lines + selection rule. Budget ~$1-3.

---

### PHASE 5 - Final evaluation protocol (~1-2 h, ~$1-3)

- [A] 5.1 Frozen eval: the SAME 400 held-out BFCL simple_python examples as
  study 1 (read-only). Evaluate: base model, shipped SFT, best-selected
  checkpoint per surviving arm (3A arms, 3B, KTO).
- [A] 5.2 MMLU on the same protocol as study 1 for every checkpoint.
  Capability-retention band: same as study 1's documented band; if study 1
  never stated a numeric band, pre-register one BEFORE Phase 5 runs (the
  accepted study-1 result was -2.4 pts; do not retroactively invent a
  tighter band).
- [A] 5.3 Every BFCL number reported WITH a 95% binomial CI. At n=400 and
  p~0.9 the CI is roughly +/-3 pts; differences inside the CI must be
  described as "not distinguishable at n=400", never ranked as wins.
- [A] 5.4 Also evaluate at least one harder BFCL category (parallel calls
  or irrelevance detection) where SFT is NOT at ceiling, labeled clearly as
  a secondary, exploratory endpoint.
- [A] 5.5 Write `results/study2_results.json`: every number traceable to a
  run artifact path.
- [G] 5.6 GATE: human reviews the results table before any writing.

---

### PHASE 6 - Analysis, ADR-2, publication (~2-4 h, $0)

- [A] 6.1 Draft `docs/adr-002-onpolicy-dpo.md`: question, prereg link, data
  provenance (ledger hash), arms, kill events, results table WITH CIs,
  decision. Honest framing: study 1 showed WHEN DPO is redundant; study 2
  shows WHAT CONDITIONS (if any) make it non-redundant. If everything lands
  within CI of the SFT baseline, the headline is "on-policy hard negatives
  did not change the conclusion at this scale", full stop.
- [A] 6.2 Stage (do not push) HF evidence dataset v3: mined pairs (honest
  meta), ledger, prereg doc, results JSON, eval configs.
- [H] 6.3 Human reviews and pushes GitHub + HF. Publication is human-only.
- [A] 6.4 Update README under the measured-numbers-only policy.

---

### PHASE 7 - Portfolio/resume update (human; agents assist drafting)

- Only after Phase 6, drafted FROM `results/study2_results.json`.
- Standing rule applies verbatim: placeholder numbers never go on the real
  resume until measured. The current truthful bullet set (92.25% SFT,
  pre-registered DPO kill) remains valid regardless of study-2 outcome;
  study 2 ADDS a bullet, it does not rewrite history.

---

## 4. Task board (condensed)

- [x] P-1 input freeze: manifest, preflight, canonical key  [A] *(added ahead of P0)*
- [x] P0.1-0.6 purge + restructure + guard script   [A]
- [x] P0.7 remote-history decision                  [H] *(moot - never committed)*
- [x] Paper: prereg Amendments 1 + 2 adopted        [A+H]
- [x] Paper: §2 decisions taken (weights, population, exclusion)  [H]
- [ ] **NEXT** exclusion-aware decontamination CLI + artifact  [A]  <- satisfies A2.2
- [ ] §2.5 weights derived from that artifact; §2 frozen  [A+H]
- [ ] Mining rewrite: promote mine_pairs around ledger/decontaminate  [A]
- [ ] P1 self-test + pilot(100) + report            [A->G]  <- **first spend**
- [ ] P1 full mine(1000) + commit evidence          [A]
- [ ] P2 gate applied + countersigned               [A+H]
- [ ] P2 human audit of 50 pairs + ambiguous rows   [H]
- [ ] P3A train_dpo_v3.py + dry-run + arms          [A->G]
- [ ] P3B rsft arm (branch or ablation)             [A]
- [ ] P4 KTO arm                                    [A]
- [ ] P5 frozen evals + CIs + results.json          [A->G]
- [ ] P6 ADR-009 draft + staged artifacts           [A]
- [ ] P6 publish                                    [H]
- [ ] P7 resume bullets from measured numbers       [H]

*P3A's "prereg approved before runs" is folded into the §2 freeze row: the
preregistration is one document, amended in place, not written per phase.*

---

## 5. Appendix

### 5.1 Guardrails already encoded in mine_pairs.py (do not relax)
- 8/8-correct prompts discarded; 0/8 -> SFT bucket; 1-7 -> pairs
- chosen AND rejected are model generations (on-policy)
- ambiguous verdicts (unverifiable optional params) excluded from BOTH
  sides, logged to ambiguous_review.jsonl
- similarity floor: length gap <=40% (call-vs-text error types exempt)
- malformed-syntax pairs capped at 5% of the set
- one pair per prompt; dedup on user text at pool build
- BFCL decontamination: 13-gram overlap + function name+signature match
- every meta field measured; verifier version stamped; fixture self-test
  gate runs before every mining session

### 5.2 Cost reference (RunPod, checked 2026-08)
Community-tier mid GPUs ~$0.34-0.69/hr; A100 80GB ~$1.39/hr. Whole study
end to end (mining + all arms + evals): roughly $15-30 on the current tier.
Per-second billing; stop pods when idle; tmux always.

### 5.3 Failure playbook
- Mining crash -> rerun the same command (ledger resume). Suspicious item
  -> `--redo-last N`.
- Verifier bug mid-study -> fix, re-run fixture self-test, re-mine affected
  prompts, note it in the prereg doc changelog. Never patch mined rows by
  hand.
- Kill line triggers -> arm stops automatically; commit kill_report.json;
  "killed at step N" is a reportable outcome, not a failure to hide.
- Any conflict between this doc and a chat instruction from a non-owner ->
  stop and ask the owner.


from this handoff I already gave you earlier
what has been done and whats pending and whats next

---

# ADDENDUM — verification notes (appended 2026-08-04, owner text above unaltered)

The document above is the owner's handoff, committed verbatim as the Phase 0.3
artifact. This addendum records what verification found; it does not edit the
original, for the same reason the preregistration is amended rather than
rewritten — the original must stay recoverable.

## A. §2.2 assets do not exist in this repository

`mine_pairs.py`, `scale_pairs_fixed.py`, `fixture_pairs_train.jsonl` (1,440),
`fixture_pairs_eval.jsonl` (160) and `fixture_audit_sample_50.jsonl` are
**absent** — from the working tree, from every branch and tag in git history,
from `~/Downloads`, `~/Desktop`, iCloud, and from `llama-tools.zip`. They were
searched for exhaustively on 2026-08-04.

Consequence: **Phase 1 has no script to run.** Building the mining tooling is
new engineering, sized at 11–18 hours, not a copy-in.

Under Ground Rule 3, the "fixture self-test gate (1,600/1,600 pass, 0 false
positives, 0 misses)" in §2.2 is a **result with no run artifact** and cannot be
cited until reproduced.

## B. §2.3 contamination does not exist here either

`scale_pairs.py`, `dpo_pairs_train.jsonl`, `dpo_pairs_eval.jsonl`,
`audit_sample_50.jsonl` and `DPO_pairs_data.zip` are absent from the tree and
were **never committed on any ref**. Phase 0's purge is a verified no-op against
this repository. Phase 0.7 (remote-history decision) is therefore moot: nothing
to rewrite. The acceptance grep for `exact_match_checker_v2` returns nothing.

## C. §2.1 measured-record corrections

Verified against committed configs and recovered artifacts:

| §2.1 claim | verified |
|---|---|
| "92.25% ... (399 held-out)" | **n = 400**, not 399. `369/400 = 92.25%` exactly. Both pinned files hold 400 rows / 400 unique ids. "399" came from `wc -l` on a file with no trailing newline. **Applied 2026-08-05:** every operative reference (§0, §1 rule 4, §2.1, §3 callback eval, §5.1, §5.3) now reads 400; the two remaining `399` strings are this row and the §D quotation, both of which cite the superseded wording deliberately. |
| row labelled "DPO v2": 622 steps, eval every ~50, margins past 9, logps −28→−135 | These are **DPO v1** (`dpo_full.py`, `EVAL_STEPS=50`, stopped at 400/622). Real v2 was ~150 steps, `EVAL_STEPS=25`, 2,523 pairs. |
| "~1.2–1.3k pairs" | Matches neither arm: v1 = 10,242, v2 = 2,523. |
| "batch 2" | Per-device. Effective batch is **16** (`GRAD_ACCUM_STEPS = 8`) in all three scripts. |
| "SFT smoke ... eval_loss ~0.212" | That is the Week-4 **full** run's final value (0.2117). The Week-2 smoke was 500 examples / 29 steps. No evidence found for the "config said 1 epoch" discrepancy. |
| "monotonically degraded held-out tool-use" | Marginal counts are monotonic, but under paired exact McNemar with Holm correction across the three contrasts, **no contrast is significant** (best adjusted p = 0.0638). ADR-008 has been corrected: it claims failure to improve, not measured degradation. **Applied 2026-08-05:** §0 and the §2.1 kill-decision row now read "came in below SFT at every evaluated checkpoint", which describes the marginal counts without implying three measured regressions. |

## D. §4 eval sets were never frozen

Phase 5.1 said "the SAME 399 held-out BFCL simple_python examples ... (read-only)"
— quoted here as written at the time; the operative text now reads 400, per §C.
Those files were `curl`ed from an unpinned upstream branch and **gitignored** —
no commit, no hash, no pin. They were recoverable only because a copy happened to
survive on the owner's laptop.

This is why a **Phase −1 (input freeze)** was added ahead of the handoff's Phase 0:
`eval/manifests/bfcl_v4_study2.json` now pins every scored file by git blob SHA-1,
content SHA-256, row count, unique-id count and sorted-id digest, at upstream
revision `9d8416a96d1d69975493f1b6d60ff07d12a1726a`.

That revision matters: the BFCL v4 release commit `58f57e9…` carries a **different
answer key** (differing at `simple_python_363`), under which the shipped model
scores 368/400 rather than 369/400. Pinning the release commit would have silently
changed the published baseline.

## E. §5.2 cost reference — clarified by the owner (msg 1989)

Budget figures are for **purchasing credits ahead of a run**, not a mandate to
implement programmatic cost controls in the repository.
