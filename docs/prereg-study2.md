# Pre-registration — Study 2 (on-policy DPO)

**Status:** OPEN. Sections below marked `[FROZEN]` are committed before the runs they
govern and may not be changed afterwards. Sections marked `[PENDING]` are not yet
committed and must be filled before the corresponding stage runs.

Governing rule: *if a threshold is not in this file before the run, it cannot be used to
select results after.*

---

## 0. Endpoint qualification probe `[FROZEN — commit this before any generation]`

### 0.1 Question

Study 1 measured tool-calling on BFCL v4 `simple_python`, where the shipped SFT model
scores **369/400 = 92.25%**. That category presents **exactly one candidate function per
item** (verified: all 400 items have `len(function) == 1`). DPO optimises *ranking among
alternatives*; a task offering a single alternative gives its preference signal nothing to
discriminate. Study 1 therefore could not separate two explanations for its negative
result:

- **(H-ceiling)** DPO is redundant because SFT is at ceiling, or
- **(H-noselect)** DPO is redundant because the task poses no selection problem.

`multiple` presents **2–4 candidate functions with one correct choice** (verified: 79 items
with 2, 85 with 3, 36 with 4) while expecting **exactly one call** — the same output shape
as `simple_python`. It changes the selection dimension and nothing else, which is why it is
the co-primary candidate.

### 0.2 What the probe measures

| | |
|---|---|
| Model | `centuriandip/llama-3.1-8b-tools-sft`, subfolder `adapter/`, revision `b6f4da479f8c6fc044ee8b802a92f47780f970c5` |
| Base | `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659` |
| Candidates | `base`, `sft` only. **No DPO checkpoint, no training.** |
| Category A | `multiple`, n = **200**, pinned in `eval/manifests/bfcl_v4_study2.json` |
| Category B | `simple_python`, n = **400**, same manifest |
| Decoding | greedy (`do_sample=False`), `max_new_tokens=512` |
| Scorer | `eval/bfcl_simple.py` — exact function-name match + per-arg value-in-accepted-list + no-extra-args |

### 0.3 Qualification rule `[FROZEN]`

> **`multiple` qualifies as study-2 co-primary endpoint if and only if shipped-SFT accuracy
> on `multiple` is ≤ 170/200 (85.0%).**

- If it qualifies: `multiple` becomes co-primary; `simple_python` becomes the
  capability-retention guardrail.
- **If it does not qualify: STOP.** Return to the owner for a new design decision. There is
  **no fallback category and no second probe** — searching further categories after seeing a
  score is outcome-driven selection and is prohibited.
- The rule is evaluated on the **shipped SFT** number only. The `base` number does not
  enter the qualification decision; it exists to produce a legitimate base→SFT lift figure,
  which the project does not currently have.

### 0.4 Reproduction check `[FROZEN]`

Category B re-scores shipped SFT on `simple_python` through the pinned inputs and the
rewritten scorer.

> **Expected: 369/400.** Any other value means the frozen pipeline disagrees with the
> published study-1 result. That is a **stop-and-report** condition, not something to be
> reconciled after the fact.

### 0.5 Handling of incomplete runs `[FROZEN]`

- A run that does not write exactly `n_prompts × n_candidates` rows is **incomplete** and
  its numbers may not be reported. `eval/bfcl_simple.py` enforces this and exits non-zero.
- Partial evidence is preserved on disk, never discarded, and never silently topped up by a
  second partial run: the script refuses to write into a non-empty output directory.
- An incomplete run may be re-run **only** into a fresh output directory, and both runs'
  manifests are retained.

### 0.6 Spend `[FROZEN]`

Estimate `$0.40–0.80`; **hard cap `$2.50`**, approved by the owner. 1,200 generations total
(200×2 + 400×2). The cap is enforced mechanically by a wall-clock ceiling in the launch
wrapper, not by intention.

---

## 1. Decontamination `[FROZEN — fixed before mined row 1]`

Screened against every set that will ever be scored, at the revisions pinned in
`eval/manifests/bfcl_v4_study2.json`:

`simple_python` · `multiple` · `irrelevance` · `live_simple`

Screens: 13-gram overlap on user text, plus exact function-name match. First match wins.

**Measured effect on the current pool** (`data/processed/sft_dedup.jsonl`, 12,160 rows):

| | count | share |
|---|---|---|
| survive | 11,263 | 92.6% |
| dropped — function-name collision | 896 | 7.4% |
| dropped — 13-gram text overlap | **1** | 0.008% |
| multi-tool prompts surviving | 7,321 of 8,117 | 90.2% |

Two things follow. The screen is **conservative by construction** — it removes prompts that
merely *offer* a tool schema an eval item also offers, which is the safe direction. And the
pool has **essentially no verbatim text contamination** with BFCL: exactly one prompt in
12,160 shares a 13-gram with any eval set.

Yield projections for the Phase 2 gate are computed on the **post-screen** pool.

---

## 2. Mining `[DRAFT — awaiting codex review and owner adoption]`

Drafting authorized by the owner, #general msg 2095. **This section is not yet frozen and
does not yet govern anything.** It becomes `[FROZEN]` only when codex has signed off on the
gate arithmetic and @dipak has explicitly adopted it, at which point it gets an adoption
line and a row in the amendments table — the same bar Amendment 2 had to clear.

Both owner decisions are now made (#general msg 2108): §2.5 weights are artifact-derived on
exact counts, and §2.7 fixes the mining population as the curated SFT pool at its cleaned
revision. What remains before freezing is mechanical: re-run decontamination over that
revision, derive the weights from the resulting artifact, and record both here.

No mining run had produced a single pair when this was drafted; no yield number of any kind
had been observed.

### 2.1 Required inputs (closes roadmap [H] 1.1)

| | |
|---|---|
| `SFT_ADAPTER` | `centuriandip/llama-3.1-8b-tools-sft`, subfolder `adapter/`, revision `b6f4da479f8c6fc044ee8b802a92f47780f970c5` |
| `BFCL_LOCAL` | the pinned fetch outputs of `eval/manifests/bfcl_v4_study2.json`, verified by `python eval/fetch_pinned_bfcl.py --verify-only` |
| Prompt pool | `data/processed/sft_dedup_v2.jsonl` — the curated SFT pool at the cleaned revision, pinned by sha256 in §2.7 |
| Verifier | `onpolicy_verifier_v1`, gated by the fixture self-test (1,600/1,600) before any prompt is mined |

The adapter revision is the one already pinned in §0.2; the same string governs both, so the
probe and the mining run cannot silently diverge.

### 2.2 Strata `[proposed]`

Every mined prompt is assigned to exactly one stratum by the count of tools its prompt
presents:

> **`multi` = 2 or more tools presented. `single` = exactly 1.**
>
> The tool list is parsed from the prompt's system message under **both** pinned formats:
> the `Tools:` JSON array, and any well-formed JSON array inside `<tools>…</tools>`. Where
> more than one well-formed list is present, **the one with the most tools wins**, so a
> prompt carrying a small example block alongside its real tool list is scored on the real
> one.
>
> A prompt is **`ineligible`** — not mined, and in neither yield term — if its tool list
> parses under neither format (`no_tool_list`), **or if it parses to zero tools**
> (`zero_tools`). A zero-tool prompt is readable and still has nothing for the model to
> call; letting it fall through to `single` would place a prompt that can never yield a
> pair into the denominator, and into the weight for a stratum it is not in. The two
> reasons are counted separately, because one is a parser gap worth fixing and the other is
> the data being what it is.
>
> Ineligible prompts are recorded in the **pre-mining eligibility artifact**
> (`mining/pool_strata.py`), never as a mining-ledger record.

Recording it in the ledger would do the opposite of what is intended: `mining/ledger.py`
treats every active non-control record as one unit of completed work, so an `ineligible`
record would land in A2.1's denominator and depress yield by exactly the prompts that were
never mined.

Both parses are required because the pool carries both formats, and a rule that reads only
one silently reclassifies the other. That is not hypothetical — see the note in §2.6.

Every ledger record carries its `stratum` label, so `y_multi` and `y_single` are recomputable
from the artifact rather than reconstructed from a run log.

### 2.3 Allocation `[proposed]`

**Proportional.** Each stratum is sampled in proportion to its target weight `w_s` (§2.5).
Both strata receive a **nonzero** allocation in the pilot and in the calibration run, so
`y_multi` and `y_single` are each estimable and neither is assumed from the other.

Allocation is planned proportionally but **realized** allocation is what gets recorded; the
gate arithmetic in §2.6 standardizes to the target weights regardless, so integer rounding
or a failed prompt cannot move the gate.

### 2.4 Sampling parameters `[proposed]`

| | |
|---|---|
| samples per prompt | 8 |
| temperature | 0.8 |
| top_p | 1.0 |
| max_new_tokens | 256 |
| seed | 20260804 |
| pilot | `--n-prompts 100`, `--out-dir mining_pilot` |
| calibration | `--n-prompts 1000`, `--out-dir mining_out` |

### 2.5 Target weights `[decided — artifact-derived, exact counts]`

Owner decision, #general msg 2108: **artifact-derived, exact counts.** The weights are not
the literal 65/35 of the earlier instruction — that figure descends from a composition count
that could read only one of the pool's two prompt formats (§2.6). The rule:

**Derived weights** (`mining/receipts/sft_dedup_v2_decontamination.json`, sha256
`fb7a0200dbeeabb831006eeb800a23d3c92d89a468666c61b098ca1277231906`, criterion
`bfcl-pool-decontamination/v1`):

> **`(n_multi, n_single, N)` = (8,173, 2,997, 11,170)**
>
> Stored as exact integers. The multi share
> 8173/11170 = 73.169% is **derived at display time and
> never stored**, so `P_std` is computed from exact ratios.

12143 cleaned source - 5 prompt-ineligible - 56 target-structural exclusions = 12082 screen inputs; of those, 912 were dropped
({'fn_name': 911, 'ngram_overlap': 1}) leaving 11,170 survivors. §1's 12,160-row figures remain
non-comparable: different source revision, different denominator.

The rule this implements:

> `w_s` = (active post-screen prompts in stratum `s`) / (all active post-screen prompts),
> computed from the **committed, hash-pinned decontamination artifact** over the mining
> prompt pool, **before any generation**, and recorded with that artifact's sha256.

The weights are a rule and not a constant on purpose. Amendment 2 (A2.2) gives authority to
the committed artifact alone, and §1's provisional composition figures have since been shown
to be wrong (§2.6). Freezing today's numbers would have frozen a defect.

If fixed design weights are ever preferred to natural composition, they may be used **only**
if labelled a design choice rather than measured composition, and recorded here before the
run.

### 2.6 Yield gate arithmetic `[proposed]`

> `y_s` = (active pairs materialized from stratum `s`) / (active post-screen prompts mined
> in stratum `s`), both terms as defined by Amendment 2 A2.1.
>
> `y_std = w_multi · y_multi + w_single · y_single`
>
> `P_std = 10,000 · y_std` — projected pairs per **10,000 post-screen active-ledger
> prompts** at the target weights.

No survival-rate conversion appears anywhere in this arithmetic. The gate is denominated in
post-screen prompts, so the decontamination survival fraction — unmeasured until A2.2's
artifact exists — never enters the decision.

**Decision table**, applied to `P_std` **unrounded**:

| `P_std` | Decision |
|---|---|
| `>= 1000` | PROCEED to Phase 3A (DPO rerun) |
| `>= 300` and `< 1000` | PROCEED CAUTIOUSLY: 1 epoch max, eval callback every ~50 steps |
| `< 300` | DO NOT run DPO. Go to Phase 3B (rejection-sampling SFT). This outcome is itself a publishable finding that strengthens the study-1 ADR |

Edges are evaluated on the exact value: `P_std = 999.6` is **not** `>= 1000`, and
`P_std = 299.99` goes to 3B. No rounding, no "approximately", at any boundary.

**This table decides on the committed calibration-run artifact only.** The 100-prompt pilot
is an *operational* gate — the human reads its histogram and approves continuing — and is
explicitly not the DPO-versus-3B scientific gate. Roadmap step 1.3 already says it: *do not
proceed on projections alone.*

**Recorded defect in §1's provisional composition.** §1 reports 8,117 multi-tool of
12,160. That count parses only the `Tools:` format; every `hermes`-format row uses
`<tools>…</tools>`, fails that parse, and was counted as not multi-tool.

Measured by `mining/pool_strata.py` over `data/processed/sft_dedup.jsonl` (sha256
`e6f4b16a606aa7846f5563d889a1d6b42bb817b7ea973c47a0f895bb5f9cbc11`), receipt committed at
`mining/receipts/sft_dedup_strata.json`:

| source | multi | single | ineligible |
|---|---:|---:|---:|
| `xlam` | 8,117 | 2,882 | 0 |
| `hermes` | 882 | 274 | 5 |
| **total** | **8,999** | **3,156** | **5** |

All 5 ineligible rows are `no_tool_list`; the pool contains **no** zero-tool
prompts (`zero_tools: 0`). The distinction is recorded because the rule must be right before
a pool that does contain them is mined, not after.

`xlam` multi-tool reproduces §1's 8,117 exactly, which is what identifies the defect rather
than merely disagreeing with it. Across both formats the share is **74.0%
of 12,155 classifiable rows, not 66.8%**. The post-screen figure, and therefore
"65/35 natural composition", inherits the same undercount.

§1's frozen text is left unaltered — A2.2 already removed its authority, and frozen text is
never edited in place. The figures above are reproducible from the committed parser and the
recorded digest rather than being a second uncommitted count.

### 2.7 Mining population `[decided — curated SFT pool, cleaned revision]`

Owner decision, #general msg 2108: **the curated SFT pool at the cleaned revision**, with
decontamination and weights re-run over it.

| | |
|---|---|
| Pool | `data/processed/sft_dedup_v2.jsonl` |
| sha256 | `9e5b7b4f3a5990b5c92e1d5f1b84d8664a9cce006f88087db7cc7219ffe76b2b` |
| Rows | 12,143 |
| Provenance | `data/clean_sft.py` over `sft_dedup.jsonl`, dropping assistant turns carrying the xLAM Python-expression annotation bug |

This supersedes §2.1's earlier `NousResearch/hermes-function-calling-v1` line: the mining
pool is the curated SFT pool, so decontamination, weights, and mining all read one
population. §1's decontamination figures do **not** carry over — they were computed over the
*uncleaned* `sft_dedup.jsonl`, and must be re-run over this revision before generation.

**Composition at this revision** (`mining/pool_strata.py`, receipt at
`mining/receipts/sft_dedup_v2_strata.json`):

| | multi | single | ineligible |
|---|---:|---:|---:|
| `xlam` | 8,102 | 2,880 | 0 |
| `hermes` | 882 | 274 | 5 |
| **total** | **8,984** | **3,154** | **5** |

**74.0% multi-tool of 12,138 classifiable.** These are
pre-screen counts; §2.5's weights come from the post-screen artifact once decontamination is
re-run here, and that artifact governs.

### 2.8 The pool's own targets are preflighted `[proposed]`

Owner instruction, #general msg 2108: extend the answer-key preflight principle to the
pool's own targets.

> Before mining, every eligible pool example is checked so that **every tool name its
> assistant turn calls appears among the tools its own system prompt presents.** A row
> failing this is a training-side `simple_python_363`: it teaches the model to invent a tool
> name, and no downstream eval attributes that habit back to the pool.
>
> **This fails closed.** A defect, or an eligible target the parser cannot read, is a
> **stop condition** for the freeze and for mining — not a row to drop. Excluding such rows
> would repair the denominator by discarding the evidence that something is wrong, which is
> the opposite of what A2.3 does with a defective answer key.

Targets are classified three ways, because "not a tool call" and "a tool call we cannot
read" are different facts:

- **`call`** — a tool call; its names are checked against the prompt.
- **`no_call`** — a target that deliberately answers in prose, e.g. asking for a missing
  argument rather than guessing one. Legitimate training signal, not a defect.
- **`unreadable`** — a target that announces itself as a tool call, or as JSON, and then
  does not parse. Under the adopted criterion `pool-target-structural-eligibility/v1` this is a **structural exclusion**, counted
  and named; absent such a criterion it is a hard failure that stops the run.

Rows whose *prompt* is ineligible (§2.2) are counted as **not applicable** rather than as
checked passes: the prompt is outside the mining population, so its target is not a target
this study will use.

**Current result at the pinned revision**, under the adopted exclusion rule
(`python -m mining.pool_strata data/processed/sft_dedup_v2.jsonl --targets`, receipt at
`mining/receipts/sft_dedup_v2_target_preflight.json`, criterion `pool-target-structural-eligibility/v1`):

| | |
|---|---:|
| raw rows | 12,143 |
| prompt-ineligible (n/a, §2.2) | 5 |
| structurally excluded (§2.8 rule) | 56 |
| **retained** | **12,082** |
| ├ call targets | 12,072 |
| └ no-call targets | 10 |
| retained name defects | 0 |
| **passed** | **True** |

Eligible 12138 = call 12072 + no_call 10 + unreadable 56; retained 12082 = eligible - unreadable; prompt_ineligible 5 is outside eligible entirely. The CLI exits 0 here and exits non-zero on any retained
name defect.

*Historical note, kept because the discovery is the reason this clause exists.* Before the
owner adopted (b), the same 56 rows were reported as `unreadable`
with `passed: False`, and §2 was correctly blocked. Adopting the exclusion did not make the
defect go away — it made the 56 rows a **declared, named output of
a versioned rule** rather than an unhandled failure. Their identities are still committed in
the receipt, and a name defect among the retained rows still fails closed.

**Owner decision, #general msg 2134: (b) preregistered exclusion.**

> **The exclusion is defined by rule, not by list.** Criterion
> **`pool-target-structural-eligibility/v1`**, recorded in the receipt:
>
> *A prompt-eligible row is structurally excluded when its assistant target announces tool-call or JSON syntax and cannot be parsed completely into the declared call form -- every tool-call marker pair accounted for, every block parsed, and every call carrying a non-empty top-level string `name`. Prose targets are `no_call` and are retained. Name mismatches are defects, not exclusions, and always fail closed.*
>
> Membership is re-derived from this criterion against whatever revision is pinned; it is
> never inherited from a frozen list of ids.
>
> The receipt records that rule's **current output**: 56 rows at
> `sft_dedup_v2.jsonl` sha256 `9e5b7b4f3a5990b5…`, all identities committed. A future
> revision recomputes membership rather than carrying these ids forward, so the list can
> never silently diverge from the rule that produced it.
>
> **The population is defined post-exclusion.** The exclusion is part of the population's
> definition, not a deletion from it — fixed before any generation, by a deterministic
> structural rule that no model output touches. §2.5's weights are derived over this
> post-exclusion, post-decontamination population, so the exclusion's systematic single-tool
> skew is *described* by the weights rather than hidden beneath them.

**Why exclusion rather than repair, recorded so the reasoning survives the decision.** The
nested `arguments.name` is a strong hint about intent, not a certainty: a row corrupt in one
place is a row whose intent is being guessed at, and 56 guessed ground truths entering the
pool as *verification keys* is synthesized provenance for a 0.46% gain.

**Inclusion would not have been neutral.** The concrete path is in the **quarantined**
`intake/quarantine/2026-08-04-chat-attachments/mine_pairs.py`, which has never been run and
must not be promoted unchanged: a target that fails call-parse is assigned
`gt = {"type": "no_call"}`, the verifier then grades any tool call against it as
`spurious_tool_call`, and a 1–7-of-8 pass split forms inverted preference pairs while 0/8
writes the false ground truth into the SFT bucket. Applied to these 56 rows that is
systematically inverted training signal — teaching the model *not* to call tools on prompts
that wanted them. This is a quarantined implementation path, not an outcome of any run. This preflight closed a live path to poisoned pairs; it did not find a cosmetic
defect. That is why §2.8's classifier distinguishes `unreadable` from `no_call` rather than
collapsing both into "not a call".

**Count reconciliation** (eligible 12138 = call 12072 + no_call 10 + unreadable 56; retained 12082 = eligible - unreadable; prompt_ineligible 5 is outside eligible entirely). Retained (12,082) exceeds
well-formed call targets (12,072) by exactly the 10
`no_call` rows, which are legitimate training data and are kept.

An earlier version of this section reported `unreadable: 0, passed: true`. That receipt was
invalid: the parser fell back to a regex search for `"name"`, which matched the *nested*
argument key and reported these 56 malformed targets as valid calls. A check that
invents the value it is checking is worse than no check, and the fallback has been removed.

### 2.9 The decontamination artifact `[built; awaiting codex review]`

Spec settled in #general msgs 2143 and 2150, recorded here so the implementer works from the
preregistration rather than from chat. **Exclusion precedes screening**, so the screen input
is 12,082:

```
12,143 cleaned source − 5 prompt-ineligible − 56 target-structural exclusions = 12,082
```

That reconciliation is stated in the artifact **before** any drop or survival count.

Criterion **`bfcl-pool-decontamination/v1`**, whose predicate pins the 13-gram user-text
normalization rule, the presented-function-name collision rule, all four manifest
categories, **the cascade order**, and fail-closed parsing.

Required contents:

- source path + sha256 (`sft_dedup_v2`); implementation identity
- eligibility criterion id + target-preflight receipt hash
- **the SHA-256 of each of the four screened `role=questions` files** from the pinned
  manifest, as `(category, local_path, sha256)` tuples — the screened question *bytes*, not
  the manifest file and not the answer keys — plus the manifest's own hash. A future BFCL
  re-pin then detectably invalidates this artifact instead of silently coexisting with it
- sorted-id digest of the 12,082 screen-input population, and of the post-screen population
- exact pre/post counts by `multi`/`single`, by drop reason, and in total
- **weights as the integer triple `(n_multi, n_single, N)`** — decimals are display-only,
  derived at render time. Exact ratios in, exact `P_std` out.

**Match order, pinned.** A row tripping both screens is attributed to whichever runs first,
so "first-match drop reason" is meaningless unless the order is part of the criterion:

> **1. 13-gram user-text overlap. 2. Exact presented-function-name collision. Stop at the
> first match.**
>
> **Category tie-break:** both screens index `gram → category` and `name → category` with
> `setdefault`, so where two categories share a gram or a function name, the **first
> category loaded from the manifest wins**, in manifest file order. That order is therefore
> part of the criterion too, and the manifest order in force is recorded in the artifact.

This matches `Decontaminator.is_contaminated` as implemented today; if the code order ever
changes, the criterion id changes with it.

A row that survives eligibility but cannot be deterministically screened is a **hard
failure**, not a new exclusion bucket.

§1's 12,160-row figures are **explicitly non-comparable**: different source revision and
different denominator.

### 2.10 Expected: near-zero spurious-call pairs `[owner-directed; pending §2 freeze]`

Recorded in advance so it is never mistaken for a miner defect. The retained population
carries only **10 genuine `no_call` rows in 12,082**, so mined **spurious-call preference
pairs will be approximately zero**. That is a property of the training pool, not of the
miner, and it is consistent with `irrelevance` being a descriptive secondary rather than an
endpoint: this pool barely exercises no-call behaviour at all.

An implementation that produced *many* spurious-call pairs from this pool would be the
anomaly worth investigating.

### 2.11 The miner re-asserts eligibility at load `[owner-directed; pending §2 freeze]`

The pool arrives pre-screened, and the miner **re-asserts that every row it receives parses
under `pool-target-structural-eligibility/v1`, refusing the run otherwise.** Checked twice,
trusted once.

The quarantined `mine_pairs.py` did the opposite: a target it could not parse silently
became a `no_call` ground truth. **That fallback is replaced by a refusal, never a
reclassification** — a row the miner cannot read is a stop condition, not a row it gets to
reinterpret.

## 3. Training arms `[PENDING]`

Arms, seeds, LoRA config, kill lines, and the callback development slice (drawn from
`live_simple`, disjoint from both final scoring sets). **Must be committed before arm 1.**

## 4. Final analysis `[PENDING]`

Paired comparison via exact McNemar on discordant items, with the pre-registered
multiplicity rule for repeated callback looks. Marginal binomial CIs are reported but do
**not** decide any comparison — every candidate is scored on the same items, so marginal CI
overlap is not a valid test of no-difference.

Power note, computed not asserted: exact McNemar reaches p<0.05 at **6 discordant items all
in one direction**; a 15/4 split gives p=0.019, while 12/5 (p=0.14), 14/7 (p=0.19) and
20/10 (p=0.099) do not. The detectable effect is therefore roughly a **3:1 win ratio among
discordant items**. Smaller or noisier effects will correctly read as *no detectable
difference* — a measured outcome, stated here in advance so it cannot later be
mischaracterised as underpowered by accident.

---

## Amendments

Any change to a `[FROZEN]` section must be recorded here with date, reason, and the commit
that made it — before the affected run starts. **Frozen text is never edited in place.** The
freeze derives its entire value from the original being recoverable, so an amendment quotes
the superseded language verbatim and supersedes it by reference.

| # | Date | Section | Change | Authorized by |
|---|---|---|---|---|
| 1 | 2026-08-04 | §0.3 | Endpoint locked unconditionally; qualification threshold demoted to a headroom gate | Owner, #general msg 1974 |
| 2 | 2026-08-05 | §0.2, §1 | Yield denominator pinned to active-ledger records; Phase 2 gate reads the committed artifact, not this document's quotation; function-name matching pinned as exact-as-presented with a fail-closed preflight | Owner, #general msg 2075 |

**Note on Amendment 2's scope.** A2.3 pins *function-name* matching only.
Argument-level matching semantics stay as `eval/bfcl_scoring.py` implements them
today, for study-1 parity. A future change to argument matching is therefore an
amendment on the same footing as a change to the name rule — not a scoring
detail — and is recorded here before the run it affects.

---

### Amendment 1 — endpoint locked a priori (2026-08-04)

**Status of data at time of adoption: no study-2 model output, evaluation result, or
probe score had been observed. No model has been run. The `multiple` baseline is
unmeasured as of this amendment.**

**Superseded text (§0.3, quoted verbatim, retained above unaltered):**

> **`multiple` qualifies as study-2 co-primary endpoint if and only if shipped-SFT accuracy
> on `multiple` is ≤ 170/200 (85.0%).**
>
> - If it qualifies: `multiple` becomes co-primary; `simple_python` becomes the
>   capability-retention guardrail.
> - **If it does not qualify: STOP.** Return to the owner for a new design decision. There is
>   **no fallback category and no second probe** — searching further categories after seeing a
>   score is outcome-driven selection and is prohibited.

**Replaced by A1.1 – A1.6 below.**

#### A1.1 `multiple` is the co-primary endpoint, unconditionally

Locked before any inference. The score does not select it; the **mechanism** does.

`simple_python` presents **exactly one** candidate function per item (`len(function) == 1`
for all 400 pinned items). DPO optimises *preference ranking between alternatives*; where
only one alternative is offered there is nothing to rank, so its signal is redundant by
construction. `multiple` presents **2–4** candidates with one correct choice
(`len(function)`: 79 items with 2, 85 with 3, 36 with 4) while expecting exactly one call —
identical output shape, one added demand.

This is the same claim ADR-008 reached from the other direction. `multiple` is therefore
not "the category with headroom"; it is the only pinned category where the causal mechanism
under test can operate at all.

**Why this is not a forking path.** `len(function)` is a **structural property of the
evaluation inputs**, fixed in files pinned by SHA-256 and git blob hash in
`eval/manifests/bfcl_v4_study2.json` before any model was run, and **independent of every
model output**. Conditioning on design facts is not conditioning on outcomes. A reader can
verify the choice could not have been outcome-dependent by counting `function` entries in
the pinned file — no model, no scores, no access to our results required.

#### A1.2 The headroom score is descriptive, never a selector

The shipped-SFT `multiple` score is still measured before Phase 2 freezes, and reported
whatever it shows. It **feeds the MDE calculation in A1.3**; it decides nothing.

`SFT > 170/200` is retained **only** as a *stop-and-consult* gate: it pauses for owner
review before further spend. It may **never** trigger a switch to a different endpoint.
There is no fallback category.

#### A1.3 Minimum detectable effect — pre-specified with discordance sensitivity

Paired power depends on the **discordant** count, not on marginal headroom alone, so a
single headroom-derived scalar would be misleading. Pre-computed grid (exact two-sided
McNemar, n = 200, α = 0.05, power ≥ 0.80):

| discordance ψ | discordant pairs | min detectable OR | implied split |
|---|---|---|---|
| 0.05 | 10 | 11.05 | 9.2 vs 0.8 |
| 0.10 | 20 | 4.00 | 16.0 vs 4.0 |
| 0.15 | 30 | 3.00 | 22.5 vs 7.5 |
| 0.20 | 40 | 2.60 | 28.9 vs 11.1 |
| 0.30 | 60 | 2.25 | 41.5 vs 18.5 |

The observed discordance is reported alongside the result, and the row it lands in states
what the study could and could not have detected — determined in advance, so a null cannot
be retrospectively excused as underpowered.

#### A1.4 Inference method, fixed in advance

- **Test:** exact two-sided McNemar on discordant pairs (exact, not chi-square — discordant
  counts here are small enough that the approximation is unreliable).
- **Interval:** **Tango's score confidence interval** for the paired difference of
  proportions, 95%, reported **regardless of direction or significance**.
- **Multiplicity:** any family of more than one confirmatory contrast is Holm-corrected, and
  both raw and adjusted p-values are reported. A contrast not pre-registered as the single
  primary comparison does not get to be reported as significant on its raw p-value.

#### A1.5 Secondary stratified analysis — pre-specified

Structural subset `len(function) in {3, 4}` of the pinned `multiple` items, where ranking
pressure is highest.

- **n = 121** (85 + 36)
- stratum id-set SHA-256: `146835ba7ff77a50e155d99f17e033a1c27e0deacc03bb00b207106fe04fcdd4`
- parent id-set SHA-256: `ce186ec8ff77e1e97325d7243bbc66b175185bcf473f05b752f969fe8d3c5241`, which
  matches the `sorted_id_sha256` already recorded for `multiple` in
  `eval/manifests/bfcl_v4_study2.json` — that existing pin is cited, not duplicated.

Both counts and both digests are re-verified from the pinned file at analysis time; a
mismatch is a stop condition.

#### A1.6 `simple_python` is the pre-registered retention secondary endpoint

Question: does DPO tuned on `multiple` degrade the already-ceilinged category? Reported with
the same paired method. A retention loss is a reportable result, not a reason to reweight
the primary.

**Net effect:** the design is strengthened, not loosened. The endpoint becomes
theory-driven rather than headroom-driven, and the possible "no detectable difference"
outcome becomes a graded result — an MDE, an interval, and a stratified contrast — rather
than a dead end.

---

### Amendment 2 — yield denominator and function-name matching (2026-08-05)

**Status of data at time of adoption: no study-2 model output, evaluation result, or
probe score had been observed. No model has been run. Both the `multiple` baseline and
the mining yield are unmeasured as of this amendment.**

**Superseded text (§1, quoted verbatim, retained above unaltered):**

> Yield projections for the Phase 2 gate are computed on the **post-screen** pool.

That sentence fixed the pool but not the arithmetic: "pairs per prompt" still had two
readings that differ by the decontamination fraction, and the fraction is unmeasured
until the artifact of A2.2 is committed. A2.1 pins both terms against the ledger.

**Superseded framing (§1's measured-effect table, retained above unaltered).** The
12,160 / 11,263 / 7,321 figures are provisional — reported from a run whose output
artifact is not yet committed. They remain as context; A2.2 removes their authority.

**Superseded text (§0.2 `Scorer` row, quoted verbatim, retained above unaltered):**

> | Scorer | `eval/bfcl_simple.py` — exact function-name match + per-arg value-in-accepted-list + no-extra-args |

"Exact function-name match" is satisfiable by a key expecting a name the model was never
offered — the `simple_python_363` defect. A2.3 keeps the rule and adds the preflight that
makes it fair.

**Replaced by A2.1 – A2.3 below.**

#### A2.1 The yield denominator is post-screen prompts

**Problem.** Fixing the mining bug where `n_prompts` was loaded *before*
decontamination (review finding 7) changes what a projected yield means.
Decontamination removes some fraction of the pool — the size of that fraction
is **not yet measured**, because the run artifact that would measure it has not
been produced or committed (see §A2.2). Whatever it turns out to be, "pairs per
prompt" is ambiguous between two denominators that differ by exactly that
fraction, and the ambiguity leans whichever way flatters the Phase 2 gate. The
denominator therefore has to be pinned now, before the number that would settle
it exists.

**Pinned definition.**

> **yield = (pairs retained by deterministic materialization from active ledger
> candidates) / (post-screen prompts mined)**
>
> The numerator counts pairs produced by re-materializing the mining ledger's
> **active** records — those not superseded by a tombstone — through the
> deterministic pair-construction path. It is not "pairs admitted", which is
> ambiguous between sampled, screened, filtered, and written; and it is not a
> count accumulated by the miner as it runs, which cannot be recovered from the
> artifact after the fact.
>
> The denominator is the count of **unique post-screen prompts bearing an
> active (non-tombstoned) outcome record in the ledger**, recomputable from the
> same artifact as the numerator. It is never the requested `--n-prompts`,
> never the pre-screen pool size, and a re-mined prompt counts once.

Two consequences follow from defining the numerator this way, and both are the
reason for it. A rolled-back mining batch cannot inflate the numerator, because
superseded records are not active. And the numerator is recomputable by anyone
holding the ledger: the same ledger materializes to the same count, so a
reported yield is checkable rather than merely reported.

The denominator is defined against the same artifact for the same reason, and
the earlier wording — "prompts the miner actually attempted" — did not survive
it. Attempts are precisely what the ledger does not preserve: a prompt that
crashes and is reconciled was attempted twice, tombstoned once, and mined once,
so counting attempts would deflate yield on exactly the runs that needed crash
recovery. That is the same class of silent lean this amendment exists to
remove, turned on the amendment's own gate number. Counting unique prompts with
an active record also keeps the two terms consistent through a rollback: a
tombstoned prompt leaves the numerator and the denominator together and returns
to both when re-mined, so yield never goes stale in one term only.

Any projection to a larger pool (e.g. "projected pairs at 10k prompts") is
computed on post-screen prompts and **must state the survival rate used for the
conversion**, so a pilot projection and a full-run result are comparable
without a silent lean in either direction. A projection that does not carry its
survival rate is not a reportable figure.

#### A2.2 The Phase 2 gate reads the run artifact, not this document

The decontamination figures quoted as context in §1 of the preregistration
(12,160 → 11,263 surviving, of which 7,321 multi-tool) are **provisional**:
they were reported from a run whose output artifact is not yet committed.

> The Phase 2 gate is evaluated against the committed decontamination run
> artifact — its path and SHA-256 — and never against this document's
> quotation of those numbers. If the committed artifact disagrees with the
> quoted figures, the artifact governs and the discrepancy is recorded as an
> amendment.

This is structural rather than remembered: quoting a provisional number inside
a preregistration is how a provisional number becomes load-bearing by accident.

#### A2.3 Function names are matched exactly, as presented

**Why this is here.** Pinning the answer key surfaced a scorer-normalization
question at `simple_python_363`, where two upstream revisions disagree on
whether the expected function name is `restaurant_search.find_closest` or
`find_closest`. That disagreement is a symptom of an unstated convention, and
the convention has to be pinned before the co-primary endpoint is scored —
otherwise study 2's headline inherits the same ambiguity at higher frequency,
discovered after the runs rather than before.

**Measured exposure** (`eval/answer_key_comparison.py`, run against the pinned
files; full report at `eval/results/answer_key_comparison.json`):

| category | rows | rows whose key uses a module-qualified name | rows where two offered tools share an unqualified tail |
|---|---:|---:|---:|
| `simple_python` | 400 | 167 (42%) | 0 (0%) |
| `multiple` | 200 | **123 (62%)** | **29 (14%)** |
| `live_simple` | 258 | 77 (30%) | 0 (0%) |

**Pinned rule.** Two parts, both executable.

> **(a) Scoring.** A predicted function name is correct if and only if it is
> **byte-identical to the answer key's name for that item**. No normalization
> is applied: no case folding, no whitespace stripping, no module-prefix
> stripping, and no matching on the unqualified tail.
>
> **(b) Preflight.** Before any item is scored, the scorer checks that every
> name the answer key expects for that item **is among the names presented to
> the model in that item's tool list**. An item that fails this check is
> refused, not graded: the run stops and names the item.
>
> **(c) Both forms offered at once.** Where a single item presents *both* a
> module-qualified name and its own bare tail as separate tools (`geometry.area`
> alongside `area`), they are **two distinct tools**, and (a) applies unchanged:
> only the key's exact string is correct. The two forms are never treated as
> spellings of one another, in either direction, and the item is not excluded.

Part (a) alone is not the rule. `parsed_name == gt_name` is satisfiable by a
key that expects a name the model was never offered — which is exactly the
`simple_python_363` defect, and exactly the case where exact matching stops
being a fair rule and becomes an unpassable one. The two parts together say
what is intended: *the presented name is the correct name.* Part (b) is what
makes that checkable rather than assumed, and it is why the rule must fail
closed instead of silently scoring such an item wrong.

Part (a) is what `eval/bfcl_scoring.py` already does (`name_ok = parsed_name ==
gt_name`). Part (b) is implemented as `preflight_key_names()` in the same
module, called by `eval/bfcl_simple.py` before generation begins, so a key
defect halts the run *before* GPU time is spent rather than after. It currently
passes on every pinned row (see below); it exists so that a future upstream
revision cannot introduce a defect that scores as a model failure.

Part (b) also runs as a **standing manifest check**: `eval/fetch_pinned_bfcl.py`
applies it to every pinned category on every fetch and every `--verify-only`
pass, so a defect is caught when the data is verified rather than only when a
category is scored. Each category is either checked or *declared* keyless in the
manifest — `irrelevance` ships no `possible_answer` file by schema — and a
category the manifest does not classify fails verification rather than being
skipped, since "the key is absent" and "no key should exist" are otherwise
indistinguishable from the file list. Receipts at the pinned revision:

| category | rows | answer-name preflight |
|---|---:|---|
| `simple_python` | 400 | clean, 400/400 key items |
| `multiple` | 200 | clean, 200/200 key items |
| `live_simple` | 258 | clean, 258/258 key items |
| `irrelevance` | 240 | no answer key by schema (declared) |

This settles the co-primary before the freeze: `multiple`, the category most
exposed to the qualified-name question at 62% qualified keys and 14% tail
collisions, carries no key defect at the pinned revision.

The amendment states the rule so it cannot drift, and records the measured
reason it must not:

- On the `multiple` co-primary, **29 of 200 items (14%) offer two or more tools
  that differ only by module prefix** — `triangle_properties.get` vs
  `circle_properties.get`, `EuclideanDistance.calculate` vs
  `angleToXAxis.calculate`. Any tail-matching rule makes those items
  **unscoreable**, because tool selection among near-identical distractors is
  precisely the capability the endpoint measures.
- Across all three pinned categories, the key name is among the names presented
  to the model in **every** row (0 exceptions in 858 rows). Exact matching is
  therefore always satisfiable; it never penalizes a model for a name the
  benchmark did not offer.
- Part (c)'s case does not currently arise: **0 of 858 rows** present both a
  qualified name and its own bare tail, and **0 rows** have a key expecting a
  bare name that is another offered tool's tail. The clause is pinned anyway,
  because it is the one configuration in which a tail-matching reader and an
  exact-matching reader would silently disagree about which tool was selected,
  and the prereg should not be the thing that has to be amended if a future
  pinned revision introduces one.

**Scope: names only.** This section pins *function-name* matching. Argument-level
matching semantics — value-in-accepted-list, the no-extra-arguments rule, the
optional-argument convention, and the numeric coercion in `values_equal()` —
remain whatever `eval/bfcl_scoring.py` implements today, which is what study 1
scored against and is deliberately unchanged for parity. That parity is the
point, and it carries a consequence worth stating rather than discovering: any
future change to argument matching is a **preregistration amendment**, not a
scoring detail, on the same footing as a change to the name rule.

**Consequence for a disagreeing key.** If a pinned answer key expects a name
that is *not* among the tools presented for that item, that item is internally
inconsistent and cannot be scored under this rule. The preflight in part (b)
raises on it, the run refuses to proceed, and the item is reported as a **key
defect** with the discrepancy documented and filed upstream. It is not silently
graded either way, and it is not dropped from the denominator without being
named. Recording the defect and re-running with a documented exclusion is a
preregistration amendment, not a scoring detail.
