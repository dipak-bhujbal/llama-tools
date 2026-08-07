# Study 2 — plan from the pilot forward

**Status:** adopted by the owner, 2026-08-07 (#general msg 2602/2604), after the
mining pilot completed. This is a *plan*, not preregistered text — nothing here
changes `docs/prereg-study2.md`, and anything that would (the KTO arm) is called
out as requiring an amendment **before** the run it affects.

## Where this starts

The mining pilot ran 2026-08-07 — the first study-2 model call, on an RTX 4090.

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

**Sequencing deviation, recorded rather than smoothed over.** §0 is
`[FROZEN — commit this before any generation]`. **The mining pilot already ran
without it.** Running the probe now **cannot retroactively satisfy "before any
generation"** — the deviation is real, it is on the record here, and the probe is
required before any *further* inference.

**What the probe is** — and it is **not** an endpoint decision:

| | |
|---|---|
| candidates | `base`, `sft` — no DPO checkpoint, no training |
| Category A | `multiple`, **n = 200** |
| Category B | `simple_python`, **n = 400** (includes the 369/400 reproduction check) |
| total | **1,200 generations** |

> **Amendment 1 supersedes original §0.3.** **A1.1 locks `multiple` as the
> co-primary endpoint unconditionally** — *"the score does not select it; the
> mechanism does"*, and there is **no fallback category**. **A1.2** keeps
> `SFT > 170/200` only as a **stop-and-consult gate**: it pauses for owner review
> before further spend and **may never trigger a switch of endpoint**. The score
> feeds A1.3's MDE calculation and decides nothing else.

**Requires its own written estimate, agent agreement, and explicit owner approval
before it runs** (WORKING-AGREEMENT §3). The earlier "~$0.15, `multiple` n=200"
line covered half the probe and is superseded by this section.

## Phase B — calibration, planning estimate about $0.60

**Written estimate, agreement between both agents, and explicit owner approval —
no approval means no spend.** Then the 1,000-prompt run (~34 min generation at
measured throughput). **§2.6 applied mechanically to that artifact:**

| `P_std` | branch |
|---|---|
| ≥ 1000 | 3A with the **frozen** arms A0–A3 available. **KTO is not among them** — it requires an adopted amendment and its own approval |
| 300–999 | **A0 only** — costs the beta sweep and KTO |
| < 300 | rejection-sampling SFT (3B) |

Owner countersigns the branch taken in one dated line.

## Phase C — training, **conditional on §2.6's branch**

**Which arms exist at all is decided by Phase B, not here:**

| branch | what may run |
|---|---|
| `P_std ≥ 1000` | **A0**, then optionally **A1–A3** |
| `300–999` | **A0 only** |
| `< 300` | **B0 only** (rejection-sampling SFT) |

**KTO is in none of these branches.** It requires an adopted amendment and its
own separate approval.

**The dev baseline precedes whichever confirmatory arm is authorized:** 258
greedy on the shipped SFT, planning estimate ~$0.10, non-negotiable — every kill
line (§3.7) and the selection rule (§3.9) compare against it.

**Every model stage here — the dev baseline and each arm — needs its own written
estimate, agent agreement, and explicit owner approval.** None is covered by an
earlier approval.

The base/shipped-SFT scoring is **not** a Phase C line: it is the §0 probe above,
at full scope, run before calibration.

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
| base → final BFCL lift | the **Phase A′ base probe** plus final scoring |
| SFT-only baseline | the comparator in the confirmatory contrast |
| beta sweep optimum | arms at β 0.1, 0.3, 0.05 — **only on a `≥1000` branch that actually launches A1–A3**, reported as exploratory |
| KTO comparison | **only if** a KTO amendment is adopted **and** that arm is separately approved and launched |

## The one thing to keep in view

**Projection, not measurement:** at the pilot's yield the pool would hold roughly
**2,000 pairs ≈ 113 optimizer steps** — about a fifth of study 1's DPO run.
**That is short, and it is acceptable.** It supports all four KPIs **only on a
branch and set of authorizations that actually launch the relevant arms** — a
`300–999` branch yields A0 alone, which answers two of them. A null is a
publishable result either way.

**If a larger sample is wanted, decide it now — before calibration runs.** The
levers are raising samples per prompt from 8 to 16, or expanding the population.
Both are amendments and both must be **adopted before the mining they affect**.

**Once calibration is observed, §2.6 applies mechanically.** A branch may not be
re-mined, re-sampled or redefined because its outcome was unwelcome. Choosing a
lever *after* seeing the yield is precisely the defect this preregistration
exists to prevent, and it would invalidate the gate it was chosen to move.
