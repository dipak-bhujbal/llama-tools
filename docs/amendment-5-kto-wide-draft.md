# Amendment 5 — wide KTO arm — **CANDIDATE DRAFT, NOT ADOPTED**

**Status: candidate draft, cycle 3.** Not in `docs/prereg-study2.md`, authorizes
nothing, and no arm may run on it. Would be folded as **record 6 / numbered
Amendment 5** only after review and explicit owner adoption.

**Relationship to Amendment 4.** A4-kto trains on the materialized pairs only, so
that A0-versus-KTO compares objectives. **This arm deliberately does the opposite**
and admits the prompts the pair miner discards. **It is a different question, not a
better version of A4**, and both may exist.

---

## Why it is worth a separate arm: the pool ceiling

**Projection from the pilot, not a calibration measurement.** Derivation, shown so
it can be checked rather than trusted:

> post-screen population **11,071** (Amendment 3 A3.3's operative weights
> `(8081, 2990, 11071)`; §2.5's `(8173, 2997, 11170)` is the superseded four-file
> artifact) · pilot `P_std = 39280300000/21820941 ≈ 1800.1194` pairs per 10,000
> prompts · **11,071 × 1800.1194 / 10,000 ≈ 1,992.9 projected pairs** · at ≤90%
> train and 16 source-pair equivalents per step, **`ceil(1793/16)` = 113 optimizer
> steps**.

**About a fifth of the 622-step study-1 DPO v1 full run** (`dpo_full.py`, stopped
at 400/622) — **not** the ~150-step DPO v2 run, which 113 steps would be most of.
**Every number above is a projection from a 100-prompt pilot, not a measurement
from a calibration artifact.**

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
inheriting §3.1 unchanged. **Runs only after A0 completes *or is killed*** —
and **A5 may never be promoted to replace A0** as the confirmatory arm, whichever
way A0 ends. Its own written estimate, agent agreement and explicit owner approval
are required.

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

### Where the solo rows actually come from

**Both solo classes are read from `mining_out/ledger.jsonl`, and only from there.**

- **8-of-8 prompts appear in no other artifact** — the miner materializes no pair
  for them, so `mined_pairs.jsonl` has no row.
- **`sft_bucket.jsonl` cannot supply the 0-of-8 completions.** Its schema is
  `{prompt_id, prompt_messages, stratum, target}` — **it stores no generations at
  all**, and `target` is the **ground-truth answer**. Taking `target` as an
  undesirable completion would train the model to **avoid the correct answer**.
  The undesirable text is the ledger's generation at the chosen index.

**Reconciliation asserted before conversion, fail-closed. `mining_summary.json`
holds histogram *counts*, not prompt IDs, so the checks differ by class:**

| class | assertion |
|---|---|
| mixed (1–7 of 8) | ledger-derived ID set **equals** `mined_pairs.jsonl`'s ID set |
| 0-of-8 | ledger-derived ID set **equals** `sft_bucket.jsonl`'s ID set |
| 8-of-8 | **no second ID artifact exists** — verified by ledger-derived count **equal to** the summary's `histogram["8"]` |
| all classes | each ledger-derived set's **size equals** the corresponding summary count |

**All four digests recorded:** `ledger.jsonl`, `mining_summary.json`,
`mined_pairs.jsonl`, `sft_bucket.jsonl`.

**Row schema for solo rows:**

| field | value |
|---|---|
| `pair_id` | `f"{prompt_id}:solo"` — **asserted disjoint** from A4's `prompt_id:chosen_index:rejected_index` form |
| `source_index` | the ledger generation index used (**0**) |
| `bucket` | `all_correct` or `zero_correct` |
| `accepted_count` | the prompt's accepted count from the ledger (8 or 0) |
| `prompt` | the ledger record's `prompt_messages`, unmodified |
| `completion` | `[{"role":"assistant","content": <ledger generation at source_index>}]` |
| `label` | `bool` — `true` for `all_correct`, `false` for `zero_correct` |
| `stratum`, `verifier_version` | copied from the ledger record; `verifier_version` asserted `onpolicy_verifier_v1` |

> **Amendment 4's assertion that "every `pair_id` appears exactly twice with
> opposite labels" is explicitly superseded for this arm**: solo `pair_id`s appear
> **exactly once**. Pair-derived `pair_id`s still appear exactly twice.

## A5.3 — the dataset is now imbalanced, and the weights must say so

At pilot composition the arm's rows are roughly **18+69 = 87 desirable** and
**18+13 = 31 undesirable** per 100 prompts — **about 2.8 : 1**. Amendment 4's
1.0/1.0 is *not* transferable; it was justified by exact balance that no longer
holds.

> **Computed from the realized *training* split, before the arm runs:**
> `desirable_weight = 1.0`;
> `undesirable_weight = n_desirable_train / n_undesirable_train` **as an exact
> rational**, converted faithfully to float and passed in that form.
>
> **The run artifact records all three:** the two counts, the exact rational, and
> the float actually passed. **A 4-decimal figure may be displayed but must never
> govern training.** **Fails if either count is zero.**
>
> Eval rows are excluded from the computation: the weights belong to the training
> objective. Choosing them by hand after seeing curves would not be auditable;
> recomputing them from the committed split is.

## A5.4 — split, ordering and leakage

**Split over prompts, not rows**, at 90/10 under §3.4's exact integer rule and
split seed 42, **then** expand — a prompt's rows must not straddle the held-out
boundary.

**§3.4's cell key does not transfer unchanged, because solo prompts have no
`rejected_reason`.** The prompt-group key for this arm is:

| class | cell key |
|---|---|
| pair prompts | `(pair, stratum, rejected_reason)` |
| 8-of-8 | `(all_correct, stratum)` |
| 0-of-8 | `(zero_correct, stratum)` |

**Thin cells refuse rather than round:** a non-empty cell that cannot yield at
least one train and one eval prompt under the exact integer rule **fails the
conversion**, rather than silently emptying an eval stratum.

**Parity is resolved *before* ordering and before the weights are computed.**

> **Within each split**, if the expanded row count is **odd**, exclude the
> **lexicographically last eligible *solo* row** whose cell remains non-empty
> afterwards. **A pair-derived row is never dropped.** The excluded row's
> `pair_id`, `prompt_id`, cell and reason are recorded in the run artifact.
> **If no eligible solo row exists, the conversion fails** rather than dropping a
> pair row or emitting an odd split.
>
> **Training weights (A5.3) and eval denominators are computed from the
> post-parity rows**, so the numbers that govern training match the rows that
> actually exist.

**Ordering under the KL constraint — an exact matching, not a retry loop.**

> 1. Sort all rows by `(pair_id, label)` ascending — a total order, no ties.
> 2. Split into `D` (desirable) and `U` (undesirable), each in that order.
> 3. Let **minority** be the smaller class and **majority** the larger. Walk the
>    minority in sorted order; each minority row is matched to the
>    **lowest-sorted unconsumed majority row carrying a different `prompt_id`**.
>    **Each majority row is consumed at most once** — it is never skipped-and-
>    reused, so no row is duplicated or omitted.
> 4. **If any minority row cannot be matched to a distinct-`prompt_id` majority
>    row, the conversion fails.** It does not silently emit a same-prompt batch,
>    which would corrupt the KL estimate.
> 5. The **even same-label surplus** of the majority class is then emitted in
>    exact sorted order, paired adjacently under the same distinct-`prompt_id`
>    rule.
> 6. Asserted: actual batch **2**, eval batch **2**, `world_size = 1`, **no
>    singleton batch reaches the KL computation**.

**The realized desirable/undesirable ratio per accumulation window is written to
the run artifact**, so a reader can see what the KL estimate actually saw.

## A5.5 — activation is decided on pre-result facts only

> **This arm may not be launched because another arm's result was disappointing.**

Launching in response to A4's or A0's **final-set** outcome would require opening
the final sets before a launch decision, which §4 forbids and which would make
every downstream contrast outcome-adaptive.

**Two admissible activation rules; the owner picks one at adoption:**

**(a) Unconditional** — the arm is simply available on the `≥1000` branch, subject
to its own estimate and approval, like A1–A3.

**(b) Pre-result threshold** — activated iff, **from the committed calibration
artifact alone**, `A0_planned_optimizer_steps < 250`, where

> `A0_planned_optimizer_steps = ceil(A0_train_pairs / 16)`, `A0_train_pairs` taken
> from the committed calibration split.

**Stated as what it is: a pre-registered short-run operational threshold, not a
power calculation.** Below 250 steps A0 gets at most five 50-step looks (§3.8),
which is a thin trajectory to select a checkpoint from and to reason about.
**Step count alone cannot establish statistical underpowering**, and this
amendment does not claim it does.

**(b) is the recommendation** — it fires exactly when the pool ceiling makes A0's
run short, is computable before anything trains, and touches no final set.

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
equality at step 0 · scoring per A4.8, **MMLU excluded** for the same §4.5 reason.

**Kill lines: §3.7 rules 1, 2, 4, 5 as A4.7 — but rule 3's replacement changes.**

> **A4.7's `eval_pair_reward_accuracy` is superseded for this arm.** It reads only
> the pair rows, which are the **minority** here, so it could kill the wide arm
> for failing at precisely the thing it is not optimising — the solo majority is
> the new signal.
>
> **Replaced by `eval_label_direction_accuracy`, over *all* held-out rows:** a row
> counts toward the numerator iff `reward > 0` for a desirable row or `reward < 0`
> for an undesirable one, **ties count false**, reward as A4.7. **First-eval
> `>= 0.99` stops the arm.** Per-row values are committed.
>
> `eval_pair_reward_accuracy` is still computed and reported, **as a diagnostic
> that kills nothing**.

## A5.8 — spend

Adds to §3.11: *one wide KTO arm — 1 epoch over the expanded set at 32 unpaired
rows per optimizer step, plus `L` dev looks × 258 greedy generations.* **Larger
than A4's arm in proportion to the row count, and it authorizes no spend.**

---

## A5.9 — no cap on the solo share

**One solo row per admitted prompt, uncapped — meaning every solo prompt enters,
except at most the pre-registered parity exclusion in A5.4.** A cap would be
another selectable design knob — its value would have to be chosen, and any choice would shape the
result. **The whole point of this arm is the full wide package**, so the
composition is **reported, not constrained**: the run artifact records the
realized desirable/undesirable and pair/solo shares.

## A5.10 — timing and run order

**This amendment must itself be reviewed and adopted before the calibration run.**
Launch eligibility is then resolved **mechanically from the calibration artifact**,
never from another arm's result.

**If both KTO arms are adopted and both become eligible, the run order is fixed
here: A0 → A4-kto → A5-kto-wide.** Fixing it in advance stops the order being
chosen once partial results exist.
