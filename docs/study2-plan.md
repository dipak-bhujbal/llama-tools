# Study 2 — plan from the pilot forward

**Status:** adopted by the owner, 2026-08-07 (#general msg 2602/2604), after the
mining pilot completed. This is a *plan*, not preregistered text — nothing here
changes `docs/prereg-study2.md`, and anything that would (the KTO arm) is called
out as requiring an amendment **before** the run it affects.

## Where this starts

The mining pilot ran 2026-08-07 — the first study-2 model call. RTX 4090,
205.8 s of generation, **$0.2131 actual**.

| | |
|---|---|
| prompts / pairs | 100 / **18** |
| histogram | `{0:13, 1:2, 2:1, 3:4, 4:2, 5:1, 6:3, 7:5, 8:69}` |
| `y_multi` / `y_single` | 0.1370 / 0.2963 |
| `y_std` / `P_std` | 0.18001 / **1800.12** (`39280300000/21820941`) |
| throughput | **2.06 s/prompt** — the first measured figure in the study |

Artifacts are committed under `mining_pilot/`. Per Amendment 2 A2.1, `P_std` is
**recomputable from `mining_pilot/ledger.jsonl`** rather than taken on trust.

**The pilot emits no `decision` field, deliberately.** §2.6 rules on the
calibration artifact; 100 prompts is an operational gate only.

## Phase A — now, effectively $0

1. **Read the 18 pilot pairs.** Each *chosen* genuinely correct, each *rejected*
   genuinely wrong rather than acceptable-but-different. This discharges most of
   the Phase 2 audit early, on 18 rows instead of ~180.
2. **Three code fixes:** per-prompt progress logging; the data transfer folded
   into the runbook (`data/processed` and `eval/bfcl_data` are git-ignored and
   were discovered missing at launch); `pod create --terminate-after` for
   calibration, where a Mac-side `sleep` timer is not an adequate backstop.
3. **Draft the KTO amendment.** §3.1 closes the arm list, so the KTO KPI
   currently has no home in the frozen text.
4. **Per-stratum split** of the 13 SFT-bucket and 69 discarded pilot prompts,
   recomputed from the ledger.

## Phase B — calibration, about $0.60

Written estimate, owner approval, then the 1,000-prompt run (~34 min generation
at measured throughput). **§2.6 applied mechanically to that artifact:**

| `P_std` | branch |
|---|---|
| ≥ 1000 | 3A with all arms available |
| 300–999 | **A0 only** — costs the beta sweep and KTO |
| < 300 | rejection-sampling SFT (3B) |

Owner countersigns the branch taken in one dated line.

## Phase C — training

**Dev baseline first: 258 greedy on the shipped SFT, ~$0.10, non-negotiable** —
every kill line (§3.7) and the selection rule (§3.9) compare against it.

Add **~$0.15** to score base and shipped SFT on `multiple` at n=200. That is what
answers the "51 to 71" question and establishes the real starting point *before*
spending on arms.

Then **A0 alone** to completion or kill, then exploratory arms and KTO, each
separately approved.

## Phase D–E

Confirmatory contrast · Holm secondaries · MMLU band · `results/study2_results.json`
· ADR-009 · HF evidence dataset · **publication by the owner** · resume bullet
drawn only from measured artifacts.

## Where the four KPIs come from

| KPI | source |
|---|---|
| base → final BFCL lift | Phase C baselines plus final scoring |
| SFT-only baseline | the comparator in the confirmatory contrast |
| beta sweep optimum | arms at β 0.1, 0.3, 0.05, reported as exploratory |
| KTO comparison | the arm added by the Phase A amendment |

## The one thing to keep in view

At the pilot's yield the entire pool holds roughly **2,000 pairs ≈ 113 optimizer
steps** — about a fifth of study 1's DPO run. **That is short, and it is
acceptable:** it measures all four KPIs, and a null is a publishable result.

If calibration comes in materially below the pilot, the two levers are **raising
samples per prompt from 8 to 16** or **expanding the population**. Both are
amendments, and both must be **decided before mining rather than after** — a
threshold chosen after seeing the yield is the defect this preregistration
exists to prevent.
