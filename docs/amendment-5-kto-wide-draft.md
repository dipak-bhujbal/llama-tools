# Amendment 5 — wide KTO arm — **CANDIDATE DRAFT, NOT ADOPTED**

**Status: candidate draft, cycle 1.** Not in `docs/prereg-study2.md`, authorizes
nothing, and no arm may run on it. Would be folded as **record 6 / numbered
Amendment 5** only after review and explicit owner adoption.

**Relationship to Amendment 4.** A4-kto trains on the materialized pairs only, so
that A0-versus-KTO compares objectives. **This arm deliberately does the opposite**
and admits the prompts the pair miner discards. **It is a different question, not a
better version of A4**, and both may exist.

---

## Why it is worth a separate arm: the pool ceiling

**Projection from the pilot, not a measurement.** At the pilot's yield the whole
11,071-prompt pool holds roughly **2,000 pairs ≈ 113 optimizer steps** — about a
fifth of study 1's DPO run.

**A null from an arm that short is hard to interpret.** "The objective doesn't
help here" and "113 steps couldn't move anything" predict the same result, and
nothing else in the current design separates them.

**What this arm changes, stated honestly and without inflation.** Admitting the
8-of-8 and 0-of-8 prompts changes **three things at once**: the objective (vs A0),
the **data composition** (correct-only and incorrect-only prompts enter), and the
**dose** (many more rows). **So it is an objective + composition + dose package.**

> **It cannot, by itself, distinguish "too short" from "inert."** A positive
> result would be consistent with more data, different data, or a longer run, and
> the write-up may not attribute it to dose alone. What it *can* do is show
> whether a substantially larger, differently-composed training set moves the
> endpoint at all — which is worth knowing and is currently unknowable.

**Pilot composition, per 100 prompts, as a traceable fact:** 18 produced pairs · 69
were 8-of-8 · 13 were 0-of-8. **Prompt coverage rises 5.56×** (100 usable prompt IDs
versus 18). **Row counts depend on A5.2's selection rule** and are not claimed here.

## A5.1 — the arm

| id | role | `loss_type` | `beta` | other |
|---|---|---|---|---|
| **A5-kto-wide** | **exploratory** | `kto` | 0.1 | unpaired objective over pairs **plus** the 8-of-8 and 0-of-8 prompts |

**Exploratory permanently**, never confirmatory or the structural secondary, no
promotion after the fact (§4.1). **Permitted only on the `P_std >= 1000` branch**,
inheriting §3.1 unchanged. Runs after A0 completes, under its own written
estimate, agent agreement and explicit owner approval.

## A5.2 — exactly which rows enter, and how many

**One row per prompt, not eight.** For each admitted prompt the **lowest-index
generation of the qualifying kind** is taken:

| source | rows contributed | label |
|---|---|---|
| pair prompts (1–7 of 8) | 2 — as Amendment 4's conversion | one `true`, one `false` |
| **8-of-8 prompts** | **1** — generation at index 0 | `true` (desirable) |
| **0-of-8 prompts** | **1** — generation at index 0 | `false` (undesirable) |

**Why one and not eight.** Eight rows from one prompt are eight near-duplicate
completions of the same input; they inflate the dataset without adding
information, and they would let a handful of prompts dominate the gradient.
**Lowest index rather than a sample keeps it deterministic and recomputable from
the ledger.**

**`pair_id` for the unpaired rows** is `f"{prompt_id}:solo"`, asserted disjoint
from Amendment 4's `prompt_id:chosen_index:rejected_index` form.

## A5.3 — the dataset is now imbalanced, and the weights must say so

At pilot composition the arm's rows are roughly **18+69 = 87 desirable** and
**18+13 = 31 undesirable** per 100 prompts — **about 2.8 : 1**. Amendment 4's
1.0/1.0 is *not* transferable; it was justified by exact balance that no longer
holds.

> **`desirable_weight` and `undesirable_weight` are computed from the committed
> calibration artifact, before the arm runs, as**
> `desirable_weight = 1.0`, `undesirable_weight = n_desirable / n_undesirable`,
> **rounded to 4 decimal places and recorded in the run artifact.**
>
> Recomputing them from the realized dataset is deterministic and auditable;
> choosing them by hand after seeing training curves would not be.

## A5.4 — split, ordering and leakage

**Split over prompts, not rows**, at 90/10 under §3.4's rule and split seed 42,
**then** expand — the same ordering Amendment 4 uses, for the same reason: a
prompt's rows must not straddle the held-out boundary.

**Ordering under the KL constraint (A4.4's problem, harder here).** With mixed
row types a balanced two-row batch is not always available. **Required:** batches
of two rows with **distinct `prompt_id`s**; where labels cannot be balanced, the
constructed order is deterministic, recorded, and **the realized
desirable/undesirable ratio per accumulation window is written to the run
artifact** so a reader can see what the KL estimate actually saw.
`world_size = 1` as A4.4.

## A5.5 — activation is decided on pre-result facts only

> **This arm may not be launched because another arm's result was disappointing.**

Launching in response to A4's or A0's **final-set** outcome would require opening
the final sets before a launch decision, which §4 forbids and which would make
every downstream contrast outcome-adaptive.

**Two admissible activation rules; the owner picks one at adoption:**

**(a) Unconditional** — the arm is simply available on the `≥1000` branch, subject
to its own estimate and approval, like A1–A3.

**(b) Pre-result threshold** — activated iff, **from the committed calibration
artifact alone**, `A0_planned_optimizer_steps < 250`. That number is computable
from mined pairs before any arm trains, touches no final set, and expresses the
real concern: **an underpowered confirmatory arm.**

**(b) is the recommendation** — it activates the arm exactly when the pool ceiling
makes A0 weak, which is the condition that motivated the arm, and it is knowable
before anything trains.

## A5.6 — family size

If both A4-kto and A5-kto-wide are adopted, §4.1's exploratory family becomes
**0 to 5** — superseding Amendment 4's `0 to 4` by reference, without editing
either.

**The price is real and compounding:** with five possible exploratory arms, every
exploratory contrast — A1, A2, A3 and both KTO arms — is Holm-corrected over up
to five. **The confirmatory contrast and the structural secondary remain families
of one and are untouched** (§4.1).

## A5.7 — everything inherited unchanged from Amendment 4

Model, revisions and LoRA from §3.4(a) · §3.4's pinned library table asserted for
exact equality · LR `5e-6`, cosine, `warmup_ratio 0.03`, `max_length 2048`, bf16,
checkpointing on · `disable_dropout=True` · `precompute_ref_log_probs=False` ·
`sync_ref_model=False` · sequential sampling · seeds 42 · ref adapter parameter-hash
equality at step 0 · **kill lines per A4.7 including `eval_pair_reward_accuracy`,
computed over the held-out *pair* rows only**, since the solo rows have no partner
to compare against · scoring per A4.8, **MMLU excluded** for the same §4.5 reason.

## A5.8 — spend

Adds to §3.11: *one wide KTO arm — 1 epoch over the expanded set at 32 unpaired
rows per optimizer step, plus `L` dev looks × 258 greedy generations.* **Larger
than A4's arm in proportion to the row count, and it authorizes no spend.**

---

## Open questions for review

1. **Is `A0_planned_optimizer_steps < 250` the right threshold**, or should it key
   on realized mined-pair count directly?
2. **Should the solo rows be capped** as a fraction of the dataset, so an
   overwhelmingly 8-of-8 pool cannot swamp the pair signal?
3. **Does `eval_pair_reward_accuracy` over pair rows only remain a meaningful kill
   line** when most training rows are solo?
