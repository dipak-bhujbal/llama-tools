# Amendment 4 — KTO comparison arm — **SUPERSEDED BY THE PREREGISTRATION**

> **This draft is no longer authoritative.** Amendment 4 was adopted by the owner
> (#general msg 2625) on the reviewed content at `d28e9e1` and is now **folded
> into `docs/prereg-study2.md` as record 5 / Amendment 4**. **Read it there.**
>
> This file is kept only as the drafting history — the four review cycles and the
> role reversal that produced the adopted text. **If the two ever disagree, the
> preregistration governs**, and this file is the one that is wrong.

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

**`pair_id` is defined, not described:**

> `pair_id = f"{prompt_id}:{chosen_index}:{rejected_index}"` — the three fields
> `mined_pairs.jsonl` already carries. Asserted unique across the dataset.

**Row schema, exact:**

| field | value |
|---|---|
| `pair_id` | as defined above — **the linkage key** |
| `prompt_id` | the source `prompt_id` |
| `prompt` | **exactly the pair's `prompt_messages` list**, unmodified |
| `completion` | **exactly** `[{"role": "assistant", "content": <chosen or rejected>}]` |
| `label` | `bool` — `true` desirable, `false` undesirable |
| `chosen_index` / `rejected_index` | both source indices, copied unchanged onto both expanded rows |
| `stratum` | the source pair's `multi` / `single` stratum, copied unchanged |
| `verifier_version` | exactly `onpolicy_verifier_v1`; any other value fails the conversion |
| `error_type` | **`= rejected_reason`** from the pair row. The miner emits `rejected_reason`; there is no `error_type` field. The complete recognized set is **`{invalid_json, missing_call, spurious_call, wrong_tool, wrong_args}`**, exactly `mining.verifier.REASONS` under `onpolicy_verifier_v1`; null or any other value fails the conversion |

**Asserted before the arm starts, fail-closed:** every `pair_id` appears exactly
twice with opposite labels; **no `pair_id` and no `prompt_id` appears in both
train and eval**; each split holds at least 2 rows. **Digests recorded:** the
input `mined_pairs.jsonl`, the split receipt, and both converted outputs.

## A4.4 — row order is part of the objective, not a detail

TRL 1.8's KTO estimates the KL term by **rotating completions within the actual
batch**, and requires `train_sampling_strategy='sequential'`. **With actual batch
2, two adjacent rows from the same pair would pair a prompt with its own
completion and destroy the mismatched-prompt KL estimate.**

> **Required, as an algorithm rather than a property.** For an ordered split of
> `N` pairs `p[0..N-1]`, **actual batch `i` = `[desirable(p_i), undesirable(p_(i+1 mod N))]`**,
> with `train_sampling_strategy='sequential'`. Every batch is balanced, carries
> two **distinct** `pair_id`s and `prompt_id`s, and there is no singleton
> remainder. **Requires `N >= 2` distinct pairs in each split.**
>
> **Pinned distributed `world_size = 1` with exactly one selected CUDA device**,
> asserted at start (`WORLD_SIZE == 1` and the accelerator reports one process).
> The physical GPU index is not pinned; the process count and batching topology
> are. Any multi-process or multi-GPU launch fails because it would change the KL
> construction silently.

## A4.5 — batch size, in the correct unit

**"Effective batch 16" from §3.5 is a *pair* unit and does not transfer** — KTO
rows are unpaired, so the same number would halve the source pairs per step and
double the step count.

> **Pair-equivalent match:** actual batch **2** × gradient accumulation **16** =
> **32 unpaired rows = 16 source-pair *equivalents* per optimizer step**, one
> epoch. *(Equivalents, not exactly 16 unique pairs: the cyclic cross-prompt
> ordering means an accumulation window need not contain 16 distinct pairs.)*
> `total_steps = ceil(train_rows / 32)` where `train_rows = 2 × train_pairs`.

## A4.6 — the rest of the configuration, fixed before the run

| | |
|---|---|
| base | `meta-llama/Llama-3.1-8B-Instruct` revision `0e9e39f249a16976918f6564b8830bc894c89659` — §3.4(a) |
| init | shipped SFT adapter revision `b6f4da479f8c6fc044ee8b802a92f47780f970c5`, trained in place — §3.4(a) |
| `beta` | 0.1 |
| `desirable_weight` / `undesirable_weight` | 1.0 / 1.0 |
| LR · schedule · warmup | **`5e-6`** · cosine · `warmup_ratio = 0.03` — §3.4(b), **not** §3.4 common, which carries none of these |
| epochs · batch | 1 epoch; per-device **2**, grad-accum **16**, eval batch 2 |
| precision | bf16, gradient checkpointing **on**, `max_length = 2048` |
| LoRA | `r = 64`, `alpha = 128`, `dropout = 0.05`, targets `["q_proj","k_proj","v_proj","o_proj"]` — §3.4(a), no new modules or rank change |
| `disable_dropout` | **`True`** — fixes runtime behavior explicitly: the adapter config retains `dropout = 0.05`, while TRL disables dropout modules during this run |
| `precompute_ref_log_probs` | **`False`** |
| `sync_ref_model` | **`False`** |
| sampling | `train_sampling_strategy='sequential'` |
| seeds | training 42, split 42 |
| libraries | `trl 1.8.0`, `peft 0.19.1`, `transformers 5.14.1`, `torch 2.13.0`, `datasets 5.0.0`, `accelerate 1.14.0` — exact equality asserted at start |
| reference model | frozen `ref` adapter copied from the shipped SFT adapter above; **parameter-hash equality asserted at step 0** before the policy updates |

**Environment scope.** Those six versions are the frozen **training-arm** pins
in §3.4, not the separate §0 inference-probe environment in
`requirements-probe.txt`. The mining pilot's recorded Torch 2.8 runtime is
therefore not evidence that a future arm satisfies §3.4, and this amendment does
not claim it is. Changing any training-arm pin requires a separate amendment to
§3.4; silently replacing the table with whatever a pod happens to provide is
forbidden.

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

> **Rule 3 is replaced by a named, defined metric — `eval_pair_reward_accuracy`.**
>
> Per-row reward: `beta * (policy_completion_logp - ref_completion_logp)`.
> For each `pair_id` in the held-out split, the pair counts toward the numerator
> iff **`reward(desirable) > reward(undesirable)` strictly** — **ties count
> false**. `eval_pair_reward_accuracy = numerator / denominator`, both committed
> alongside the per-pair rows. **First-eval `>= 0.99` stops the arm**, the same
> threshold rule 3 uses.

**A missing or renamed metric fails before training or eval begins**, never
silently disabling a kill line. **The kill report records** numerator,
denominator, rate, threshold, the split digest, the look index and the optimizer
step.

## A4.8 — scoring and analysis

**Scored only after every launched arm's checkpoint is selected**, on `multiple`
(n = 200) and `simple_python` (n = 400). **Final sets may not influence launch or
checkpoint selection.** Its `multiple` contrast versus shipped SFT joins the
**exploratory Holm family**; `simple_python` retention stays a **band outside any
test family** (§4.1a).

**MMLU is deliberately excluded.** Frozen §4.5 runs MMLU *"only for a candidate
that clears the primary contrast"*, and **A4-kto is permanently exploratory, so
it can never clear it.** Including MMLU here would have required amending §4.5
and paying ~90 minutes per candidate for a check the frozen text does not
authorise for this arm.

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
(msg 2625), because a family size settled **before the calibration artifact and its yield
are observed** cannot be argued about afterwards. *(The pilot's yield is already
observed; the calibration yield is the one that matters here.)*

## A4.11 — spend

Adds to §3.11: *one KTO arm — 1 epoch of KTO at 32 unpaired rows per optimizer
step over the converted pair set, plus `L` dev looks × 258 greedy generations
(`L` per §3.8: cadence `max(50, ceil(total_steps/20))`, `L_max = 20`)*.

**Adoption authorizes no spend.**

---

## To adopt

1. @codex review
2. owner adoption of the **final reviewed text**
3. folded in as record 5 / Amendment 4, with §4.1's superseded sentence **quoted
   verbatim and superseded by reference**, never edited in place
4. the record-vs-amendment numbering note extended so "Amendment 4" resolves
