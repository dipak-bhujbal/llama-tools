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
| cost | **$0.2131 provider-reported** (RunPod balance delta) — **no committed cost receipt exists**, so this is not a measured figure under the evidence rule |

Artifacts are committed under `mining_pilot/`. Per Amendment 2 A2.1, `P_std` is
**recomputable from `mining_pilot/ledger.jsonl`** rather than taken on trust.

**The pilot emits no `decision` field, deliberately.** §2.6 rules on the
calibration artifact; 100 prompts is an operational gate only.

## Phase A — now, effectively $0

1. **Read the 18 pilot pairs.** Each *chosen* genuinely correct, each *rejected*
   genuinely wrong rather than acceptable-but-different. **This is a preliminary
   qualitative check and discharges no part of §2's required 50-pair calibration
   audit** — that audit is over the calibration artifact, which does not exist
   yet. Reading these early is cheap insurance, not credit against a later gate.
2. **Three code fixes:** per-prompt progress logging; the data transfer folded
   into the runbook (`data/processed` and `eval/bfcl_data` are git-ignored and
   were discovered missing at launch); `pod create --terminate-after` for
   calibration, where a Mac-side `sleep` timer is not an adequate backstop.
3. **Draft the KTO amendment.** §3.1 closes the arm list, so the KTO KPI
   currently has no home in the frozen text.
4. **Per-stratum split** of the 13 SFT-bucket and 69 discarded pilot prompts,
   recomputed from the ledger.

## Phase A′ — the §0 endpoint qualification probe, **overdue**

§0 is `[FROZEN — commit this before any generation]` and has **not run**. It is
not a Phase C line item and it is not `multiple`-only:

| | |
|---|---|
| candidates | `base`, `sft` — **no DPO checkpoint, no training** |
| Category A | `multiple`, **n = 200** |
| Category B | `simple_python`, **n = 400** (includes the 369/400 reproduction check) |
| total | **1,200 generations** |

> **Qualification rule (§0.3):** `multiple` qualifies as a study-2 co-primary
> endpoint **iff shipped-SFT accuracy on `multiple` is ≤ 170/200 (85.0%)**.

**This decides whether the study's endpoint is valid at all**, so it belongs
ahead of calibration and ahead of any further inference. **It needs its own
written estimate, agreement, and explicit owner approval before it runs** — the
earlier "~$0.15, `multiple` n=200" line covered half the frozen probe and is
superseded by this section.

## Phase B — calibration, planning estimate about $0.60

Written estimate, owner approval, then the 1,000-prompt run (~34 min generation
at measured throughput). **§2.6 applied mechanically to that artifact:**

| `P_std` | branch |
|---|---|
| ≥ 1000 | 3A with the **frozen** arms A0–A3 available. **KTO is not among them** — it requires an adopted amendment and its own approval |
| 300–999 | **A0 only** — costs the beta sweep and KTO |
| < 300 | rejection-sampling SFT (3B) |

Owner countersigns the branch taken in one dated line.

## Phase C — training

**Dev baseline first: 258 greedy on the shipped SFT, planning estimate ~$0.10,
non-negotiable** —
every kill line (§3.7) and the selection rule (§3.9) compare against it.

The base/shipped-SFT scoring that answers the "51 to 71" question is **not** a
Phase C line — it is the §0 probe above, at full scope, run before calibration.

Then **A0 alone** to completion or kill, then exploratory arms and KTO, each
separately approved.

## Phase D–E

Confirmatory contrast · structural secondary · MMLU band · `results/study2_results.json`
· ADR-009 · HF evidence dataset · **publication by the owner** · resume bullet
drawn only from measured artifacts.

**§4.1's multiplicity rule, stated correctly:** the one confirmatory contrast and
the one structural secondary are each **families of one and take no adjustment**.
**Holm applies only across exploratory-arm contrasts actually launched**, over
exactly the number launched, recorded before the analysis runs.

## Where the four KPIs come from

| KPI | source |
|---|---|
| base → final BFCL lift | Phase C baselines plus final scoring |
| SFT-only baseline | the comparator in the confirmatory contrast |
| beta sweep optimum | arms at β 0.1, 0.3, 0.05, reported as exploratory |
| KTO comparison | the arm added by the Phase A amendment |

## The one thing to keep in view

**Projection, not measurement:** at the pilot's yield the pool would hold roughly
**2,000 pairs ≈ 113 optimizer steps** — about a fifth of study 1's DPO run.
**That is short, and it is acceptable:** it measures all four KPIs, and a null is
a publishable result.

**If a larger sample is wanted, decide it now — before calibration runs.** The
levers are raising samples per prompt from 8 to 16, or expanding the population.
Both are amendments and both must be **adopted before the mining they affect**.

**Once calibration is observed, §2.6 applies mechanically.** A branch may not be
re-mined, re-sampled or redefined because its outcome was unwelcome. Choosing a
lever *after* seeing the yield is precisely the defect this preregistration
exists to prevent, and it would invalidate the gate it was chosen to move.
