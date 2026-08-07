# Amendment 4 — KTO comparison arm — **DRAFT, NOT ADOPTED**

**Status: draft for review, cycle 2.** Not in `docs/prereg-study2.md`, authorizes
nothing, and no arm may run on it. To be folded in as **record 5 / numbered
Amendment 4** once reviewed and adopted in its final reviewed form.

**Owner decisions already recorded (#general msg 2625) and preserved unchanged
through this revision:** A4.1 pairs-only · A4.3 equal weights · A4.5 Holm
widening accepted · adoption before calibration.

**Why it exists.** Frozen §3.1: *"An arm not listed in §3.5 or §3.6 requires an
amendment before it runs. That includes the KTO arm sketched at roadmap Phase 4."*

---

## A4.1 — KTO trains on exactly the materialized pairs, converted

Each pair contributes one **desirable** and one **undesirable** row. The 8-of-8
and 0-of-8 prompts are **not** admitted here.

**Reasoning:** A0-versus-KTO compares *objectives*. Admitting the unpaired
prompts changes data volume **and** composition at once, so a difference could
not be attributed to the objective. **That wider design is a separate candidate
amendment** (`docs/amendment-5-kto-wide-draft.md`), not a variant of this one.

**Scale, stated only as a traceable pilot fact and only as a planning
illustration:** the pilot mined **100 prompt IDs**, of which **18 produced
pairs** — **5.56× prompt coverage**. It is *not* a claim about row counts, which
depend on a row-selection rule this amendment does not define.

## A4.2 — the arm and where it is permitted

| id | role | `loss_type` | `beta` | other |
|---|---|---|---|---|
| **A4-kto** | **exploratory** | `kto` | 0.1 | unpaired objective over the converted pair set |

**Permitted only on the `P_std >= 1000` branch** — §3.1 already bars exploratory
arms from the `300–999` branch, and this amendment widens no branch. **Run order
unchanged:** A0 first, alone, to completion or kill. Its own written estimate,
agent agreement and explicit owner approval are required before it runs.

**A4-kto is exploratory permanently.** It may never be the confirmatory candidate
or the structural secondary, and it cannot be promoted after the fact (§4.1).

## A4.3 — split first, then expand; the dataset contract is pinned

**Order is load-bearing.** The 90/10 split runs over **pairs**, under §3.4's exact
integer rule, track-specific cell key and **split seed 42** — *then* each pair
expands into its two rows. Splitting after expansion would place a prompt's
desirable and undesirable rows on opposite sides and leak the held-out set.

**Row schema:**

| field | |
|---|---|
| `pair_id` | stable identifier of the source pair — **the linkage key**, not `prompt_id` alone |
| `prompt_id` | the ledger prompt it derives from |
| `prompt` | the exact conversational turns preceding the target, roles intact |
| `completion` | the assistant completion (chosen or rejected) |
| `label` | `true` = desirable, `false` = undesirable |
| `error_type` | carried through for the split cell key |

**Asserted before the arm starts, fail-closed:** every `pair_id` appears exactly
twice with opposite labels; **no `pair_id` and no `prompt_id` appears in both
train and eval**; each split holds at least 2 rows. **Digests recorded:** the
input `mined_pairs.jsonl`, the split receipt, and both converted outputs.

## A4.4 — row order is part of the objective, not a detail

TRL 1.8's KTO estimates the KL term by **rotating completions within the actual
batch**, and requires `train_sampling_strategy='sequential'`. **With actual batch
2, two adjacent rows from the same pair would pair a prompt with its own
completion and destroy the mismatched-prompt KL estimate.**

> **Required:** deterministic batch construction in which the two rows of any
> actual batch carry **distinct `prompt_id`s** and balanced labels; sequential
> sampling; **single-GPU, fail-closed** if more than one device is visible, since
> multi-device batching would silently change the KL construction.

## A4.5 — batch size, in the correct unit

**"Effective batch 16" from §3.5 is a *pair* unit and does not transfer** — KTO
rows are unpaired, so the same number would halve the source pairs per step and
double the step count.

> **Pair-equivalent match:** actual batch **2** × gradient accumulation **16** =
> **32 unpaired rows = 16 source pairs per optimizer step**, one epoch.
> `total_steps = ceil(train_rows / 32)` where `train_rows = 2 × train_pairs`.

## A4.6 — the rest of the configuration, fixed before the run

`beta` 0.1 · `desirable_weight` 1.0 · `undesirable_weight` 1.0 · training seed 42
· split seed 42 · bf16 · LR, scheduler and warmup **as §3.4's common settings** ·
max sequence length as §3.4 · gradient checkpointing and dropout as §3.4 ·
`precompute_ref_log_probs` recorded · **`sync_ref_model=False`** · library
versions pinned in the run artifact · **asserted: the reference model is the
shipped SFT adapter**.

**Why both weights are 1.0:** A4.1's conversion is **exactly balanced**, and 1/1
is the principled setting for that regime. *(Not "because TRL's defaults were
designed for imbalance" — they were not; the earlier draft said so and was
wrong.)*

**What matching `beta` does and does not buy.** It removes one difference. It does
**not** make the objective the only difference: **KTO's unpaired loss and its
in-batch KL construction differ mechanically from DPO's paired loss even on
identical source pairs.** The contrast is *objective-package versus
objective-package*, and the write-up may not describe it as an isolated change of
loss function.

## A4.7 — kill lines, inherited selectively rather than wholesale

| §3.7 rule | under KTO |
|---|---|
| 1 — non-finite | **applies unchanged** |
| 2 — `eval_rewards/chosen < -0.25` | **applies** — KTO emits a chosen reward |
| 3 — first-eval `eval_rewards/accuracies >= 0.99` | **does not exist under KTO.** TRL's KTO trainer emits no `eval_rewards/accuracies` |
| 4 — dev health, absolute item margin | **applies unchanged** |
| 5 — direction guard | **applies unchanged** |

> **Rule 3 is replaced, not dropped:** the equivalent easy-data check is the
> **pair-linked rate at which the desirable row's reward exceeds its own pair's
> undesirable row, over the held-out pair split**, at the first eval, with the
> same `>= 0.99` threshold. `pair_id` is what makes this computable — which is
> why A4.3 makes it the linkage key.

**The exact emitted metric names and the kill-report fields are pinned in the run
artifact**, so a renamed metric fails closed instead of silently disabling a kill
line.

## A4.8 — scoring and analysis

**Scored only after every launched arm's checkpoint is selected**, on `multiple`
(n = 200), `simple_python` (n = 400) and MMLU. **Final sets may not influence
launch or checkpoint selection.** Its `multiple` contrast versus shipped SFT
joins the **exploratory Holm family**; retention and MMLU remain **bands outside
any test family** (§4.1a, §4.5).

## A4.9 — exploratory family size

Frozen §4.1 reads:

> *"Family size = the number of exploratory arms launched (0 to 3), recorded
> before the analysis runs"*

**Superseded to `0 to 4`**, Holm-corrected over exactly the count launched.

**The price, stated for weighing:** a fourth possible exploratory arm widens the
correction for A1–A3, which have nothing to do with KTO. **Accepted by the owner
(msg 2625)** on the grounds that exploratory contrasts can never be promoted to
the result, so what is lost is sensitivity in reporting which exploratory
contrasts look individually notable — **not any part of the confirmatory
verdict**, which §4.1 keeps as a family of one taking no adjustment.

## A4.10 — timing

**This amendment proposes that its own adoption precede the calibration run.**

**Stated honestly: current §4.1 does not require this** — it defines family size
from the arms actually launched, recorded before the analysis. **This is a
stricter rule this amendment adopts for itself**, on the owner's instruction
(msg 2625), because a family size settled before any yield is observed cannot be
argued about afterwards.

## A4.11 — spend

Adds to §3.11: *one KTO arm — 1 epoch of KTO at 32 unpaired rows per optimizer
step over the converted pair set, plus `L` dev looks × 258 greedy generations
(`L` per §3.8: cadence `max(50, ceil(total_steps/20))`, `L_max = 20`)*.

**Adoption authorizes no spend.**

---

## To adopt

1. @codex review 2. owner adoption of the **final reviewed text**
3. folded in as record 5 / Amendment 4, with §4.1's superseded sentence **quoted
verbatim and superseded by reference**, never edited in place
4. the record-vs-amendment numbering note extended so "Amendment 4" resolves
