# Amendment 4 — KTO comparison arm — **DRAFT, NOT ADOPTED**

**Status: draft for review.** Not adopted, not in `docs/prereg-study2.md`, and it
authorizes nothing. Frozen text is never edited in place; this is written to be
folded in as **record 5 / numbered Amendment 4** once @codex has reviewed it and
the owner has adopted it. No arm may run on it before then.

**Why it exists.** Frozen §3.1: *"An arm not listed in §3.5 or §3.6 requires an
amendment before it runs. That includes the KTO arm sketched at roadmap Phase 4."*
The KPI "KTO comparison" therefore has no home in the frozen text today.

---

## The one decision that needs the owner, and it is not a detail

**KTO's actual advantage is that it consumes *unpaired* data.** The pilot shows
what that would mean here, per 100 prompts:

| pilot outcome | count | usable by DPO? | usable by KTO? |
|---|---:|---|---|
| 1–7 of 8 correct → **pair** | 18 | yes | yes |
| 8 of 8 correct → discarded | 69 | **no** | **yes** — desirable-only |
| 0 of 8 correct → SFT bucket | 13 | **no** | **yes** — undesirable-only |

**So a KTO arm could train on roughly five times the data A0 sees.** That is a
real option and it is *not* the option this draft takes.

> **A4.1 — KTO trains on exactly the materialized pairs, converted. Nothing else.**
>
> Each pair contributes one **desirable** (`chosen`) and one **undesirable**
> (`rejected`) example, carrying `prompt_id` for linkage. The 8-of-8 and 0-of-8
> prompts are **not** admitted.

**Reasoning, stated so a reader can disagree with it deliberately.** A0 versus KTO
is meant to compare *objectives*. If KTO also gets 5× the examples, the contrast
confounds algorithm with data volume and answers neither question. **Admitting the
unpaired prompts is a legitimate and interesting arm — it is just a different one,
and it needs its own amendment rather than arriving inside this one.**

**Consequence to accept knowingly:** this makes the KTO arm's dataset perfectly
balanced (equal desirable and undesirable), which is unusual for KTO and is the
regime its `desirable_weight`/`undesirable_weight` defaults were *not* designed
for. A4.3 pins those weights explicitly rather than inheriting a default tuned for
imbalanced data.

---

## A4.2 — the arm, and where it is permitted

Adds one row to §3.5's table:

| id | role | `loss_type` | `beta` | other |
|---|---|---|---|---|
| **A4-kto** | **exploratory** | `kto` | 0.1 | unpaired objective over the converted pair set |

**Permitted only on the `P_std >= 1000` branch.** Frozen §3.1 already states that
the `300–999` branch runs **A0 only** and that *"no exploratory arm may run on
this branch"*; A4-kto is exploratory and inherits that restriction unchanged.
**This amendment does not widen any branch.**

**Run order unchanged:** A0 first, alone, to completion or kill. A4-kto runs after
A0 has finished and its result is recorded, under **its own written estimate and
explicit owner approval** (§3.1, WORKING-AGREEMENT §3).

**Everything else is inherited, not restated:** kill lines §3.7, look cadence
§3.8 (every 50 optimizer steps), checkpoint selection §3.9 (eligibility
`n_base − 2`, tie set K = 3 items, most-steps tie-break), and the §3.3 dev
baseline it is measured against.

## A4.3 — hyperparameters, fixed before the run

| | |
|---|---|
| `loss_type` | `kto` |
| `beta` | **0.1** — matched to A0 so the objective is the only difference |
| `desirable_weight` | **1.0** |
| `undesirable_weight` | **1.0** |
| epochs / batch | 1 epoch, effective batch 16 — as §3.5's 3A arms |

**Why both weights are 1.0:** A4.1 produces exactly one desirable and one
undesirable per pair, so the dataset is balanced by construction and no
reweighting is warranted. Stating them explicitly stops a library default from
silently becoming a study parameter.

## A4.4 — the conversion is deterministic and recorded

The converted dataset is built from `mining_out/mined_pairs.jsonl` by a committed
script, is **content-addressed by sha256 before the arm starts**, and records the
`prompt_id` linkage for every row so any KTO example can be traced back to the
pair and the ledger record it came from. **A run whose converted-dataset digest
does not match the recorded one refuses to start**, on the same fail-closed
footing as §2.11 and §3.3.

## A4.5 — this changes the exploratory family size, and that is the real cost

Frozen §4.1 reads:

> *"Family size = the number of exploratory arms launched (0 to 3), recorded
> before the analysis runs"*

**Superseded to:** *family size = the number of exploratory arms launched, **0 to
4**, recorded before the analysis runs*, Holm-corrected over exactly that count.

**This is the honest price of the amendment and it should be weighed before
adoption:** adding a fourth possible exploratory arm **widens the Holm correction
for every exploratory arm**, including A1–A3. If all four launch, each exploratory
contrast is corrected over four rather than three. **The confirmatory contrast and
the structural secondary are untouched** — §4.1 makes each a family of one taking
no adjustment, and this amendment does not change that.

## A4.6 — spend

Adds to §3.11's countable-work table:

| stage | countable work |
|---|---|
| one KTO arm | 1 epoch of KTO at effective batch 16 over the converted pair set, plus `L` dev looks × 258 greedy generations (`L` per §3.8) |

**Adoption authorizes no spend.** The arm needs its own written estimate, agent
agreement, and explicit owner approval before it runs.

---

## What adoption would require, in order

1. @codex reviews this draft
2. Owner adopts it explicitly in `#general`
3. It is folded into `docs/prereg-study2.md` as **record 5 / Amendment 4**, with
   §4.1's superseded sentence **quoted verbatim** and superseded by reference —
   never edited in place
4. The amendments-table note about record-vs-amendment numbering is extended, so
   a citation of "Amendment 4" still resolves

**Adoption must precede the calibration run if the KTO arm is wanted**, because
A4.5 changes the multiplicity family that the analysis is corrected over, and a
family size chosen after seeing results is the defect this preregistration exists
to prevent.
