# Pre-registration — Study 2 (on-policy DPO)

**Status:** OPEN. Sections below marked `[FROZEN]` are committed before the runs they
govern and may not be changed afterwards. Sections marked `[PENDING]` are not yet
committed and must be filled before the corresponding stage runs.

**As of 2026-08-07 every section is `[FROZEN]`** — §0 and §1 before mining was designed, §2 on
2026-08-05, and §3 and §4 owner-adopted on 2026-08-06 and frozen on 2026-08-07 when this exact-
content commit cleared the required review — and all four amendment records are adopted. The
document was completed **before any study-2 model output, evaluation result, probe score, mined
pair, or yield number existed**, at $0 of model spend. Everything after this point is execution
against text that can no longer move without an amendment.

**Freeze/review ordering.** When the owner authorizes adoption before the required exact-
content review is complete, the adoption status may be recorded immediately, but the freeze
attaches to the first exact content commit that clears that review. Until sign-off, correcting
review findings completes the adoption package rather than amending operative text. After
sign-off, every content change requires an amendment under the rule below. This keeps an
unreviewed status-label commit from freezing defects that were already in review.

**Audit note — premature freeze label, 2026-08-06.** Codex's cycle-2 review reported the §3.7
scope defect and §3.9 fractional-to-integral inference at 13:51, before commit `f0e9f79`
applied the `[FROZEN]` labels at approximately 13:54 without incorporating those findings.
The stale §3.2 selection consequence was identified after that commit. No study-2 run or
model call occurred, and `f0e9f79` never reached `origin`. The wording corrections in the
first review-cleared exact-content commit complete the adoption under the ordering rule above;
they do not create an amendment record. The 2026-08-07 freeze date is prospective in this
candidate commit and becomes effective only at exact-content sign-off; if sign-off occurs on
a later date, the date must be corrected before the freeze attaches.

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

## 2. Mining `[FROZEN — adopted by the owner 2026-08-05, #general msg 2181]`

Drafting authorized by the owner, #general msg 2095. The **decontamination artifact
package** was signed off at `0dccddd` (#general msg 2169); that sign-off covered the
implementation and artifact, and explicitly left §2 as a draft. **Codex signed off the full
candidate content at `6110545` (#general msg 2173); the owner adopted it at #general msg
2181.** This section now governs:
it became `[FROZEN]` on the owner's explicit adoption, recorded in the adoption line below
and as amendments-table row 3 — the same bar Amendment 2 cleared.

> **Adoption line.** Adopted by the owner on 2026-08-05 (#general msg 2181):
>
> *Adopt §2 Mining as frozen in reviewed content commit `d6ff0cad0aae952f2d4ce2e5e066f36d73d98dcd`,
> published unchanged on public/private `main` by merge commit
> `d7c4e5cc76fdb116c75e7b93c77d538c3bde5b53`, before any study-2 model call. The earlier
> provisional 65/35 composition came from a parser that read only one tool-list format; the
> frozen artifact-derived weights are `(8173, 2997, 11170)` (73.169% / 26.831%) over the
> corrected post-exclusion, post-screen population.*
>
> No study-2 model output had been observed at adoption. Adoption authorized no model call
> and no spend.

All owner decisions are made and all mechanical work is done: the decontamination artifact
is built and committed, the weights are derived from it as an exact integer triple, and
codex verified every binding independently, and the owner adopted the reviewed package in
#general msg 2181.

No mining run had produced a single pair when this was drafted; no yield number of any kind
had been observed.

### 2.1 Required inputs `[FROZEN]` (closes roadmap [H] 1.1)

| | |
|---|---|
| `SFT_ADAPTER` | `centuriandip/llama-3.1-8b-tools-sft`, subfolder `adapter/`, revision `b6f4da479f8c6fc044ee8b802a92f47780f970c5` |
| `BFCL_LOCAL` | the pinned fetch outputs of `eval/manifests/bfcl_v4_study2.json`, verified by `python eval/fetch_pinned_bfcl.py --verify-only` |
| Prompt pool | `data/processed/sft_dedup_v2.jsonl` — the curated SFT pool at the cleaned revision, pinned by sha256 in §2.7 |
| Verifier | `onpolicy_verifier_v1`, gated by the fixture self-test (1,600/1,600) before any prompt is mined |

The adapter revision is the one already pinned in §0.2; the same string governs both, so the
probe and the mining run cannot silently diverge.

### 2.2 Strata `[FROZEN]`

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

### 2.3 Allocation `[FROZEN]`

**Proportional.** Each stratum is sampled in proportion to its target weight `w_s` (§2.5).
Both strata receive a **nonzero** allocation in the pilot and in the calibration run, so
`y_multi` and `y_single` are each estimable and neither is assumed from the other.

Allocation is planned proportionally but **realized** allocation is what gets recorded; the
gate arithmetic in §2.6 standardizes to the target weights regardless, so integer rounding
or a failed prompt cannot move the gate.

### 2.4 Sampling parameters `[FROZEN]`

| | |
|---|---|
| samples per prompt | 8 |
| temperature | 0.8 |
| top_p | 1.0 |
| max_new_tokens | 256 |
| seed | 20260804 |
| pilot | `--n-prompts 100`, `--out-dir mining_pilot` |
| calibration | `--n-prompts 1000`, `--out-dir mining_out` |

### 2.5 Target weights `[FROZEN — artifact-derived, exact counts]`

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

### 2.6 Yield gate arithmetic `[FROZEN]`

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

### 2.7 Mining population `[FROZEN — curated SFT pool, cleaned revision]`

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

### 2.8 The pool's own targets are preflighted `[FROZEN]`

Owner instruction, #general msg 2108: extend the answer-key preflight principle to the
pool's own targets.

> Before mining, every eligible pool example is checked so that **every tool name its
> assistant turn calls appears among the tools its own system prompt presents.** A row
> failing this is a training-side `simple_python_363`: it teaches the model to invent a tool
> name, and no downstream eval attributes that habit back to the pool.
>
> **Fail-closed rule, as adopted.** A **retained name mismatch always stops** the freeze and
> mining: it is a claim about a row we can read, and dropping it would repair the denominator
> by discarding evidence, which is the opposite of what A2.3 does with a defective answer
> key. An **unreadable target is excluded only under the owner-adopted, versioned criterion
> `pool-target-structural-eligibility/v1`**, which names it, counts it, and commits its
> identity; **absent that criterion it stops the run.**

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

### 2.9 The decontamination artifact `[FROZEN — artifact signed off at 0dccddd]`

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

### 2.10 Expected: near-zero spurious-call pairs `[FROZEN — owner-directed]`

Recorded in advance so it is never mistaken for a miner defect. The retained population
carries only **10 genuine `no_call` rows in 12,082**, so mined **spurious-call preference
pairs will be approximately zero**. That is a property of the training pool, not of the
miner, and it is consistent with `irrelevance` being a descriptive secondary rather than an
endpoint: this pool barely exercises no-call behaviour at all.

An implementation that produced *many* spurious-call pairs from this pool would be the
anomaly worth investigating.

### 2.11 The miner re-asserts eligibility at load `[FROZEN — owner-directed]`

The pool arrives pre-screened, and the miner **re-asserts that every row it receives parses
under `pool-target-structural-eligibility/v1`, refusing the run otherwise.** Checked twice,
trusted once.

The quarantined `mine_pairs.py` did the opposite: a target it could not parse silently
became a `no_call` ground truth. **That fallback is replaced by a refusal, never a
reclassification** — a row the miner cannot read is a stop condition, not a row it gets to
reinterpret.

## 3. Training arms `[FROZEN — owner-adopted 2026-08-06; exact-content review cleared 2026-08-07]`

> **Adoption line.** Adopted by the owner on 2026-08-06 (#general msg 2297):
>
> *"Confirmed: keep the existing absolute margin unchanged. No McNemar veto is created.
> Eligibility stays `n_ckpt >= n_base − 2` on the 258-item subset, and McNemar remains a
> recorded diagnostic that triggers nothing. […] the existing absolute rule, unchanged. K = 3
> stands as pinned. Nothing else in my Decision A text needs amending. Adopt §3 and §4,
> publish Amendment 3, unblock the miner, and send me the pilot estimate when it exists."*
>
> **No study-2 model output, evaluation result, probe score, mined pair, or yield number had
> been observed at adoption. No model had been run and model spend was $0.** Adoption
> authorized no model call and no spend; §3.11 still authorizes none.

This section now governs. Its content reached adoption through the review record: drafted
2026-08-06, reworked against three rounds of reviewer findings, reshaped by owner Decision C
(#general msg 2244) and owner Decision A (#general msg 2292), and cleared for adoption by the
confirmation quoted above. **Frozen text is never edited in place** — any later change is an
amendment, recorded in the table below with its date, reason, and commit, before the run it
affects.

**Status of data at drafting: no study-2 model output, evaluation result, probe score,
mined pair, or yield number had been observed. No model had been run.**

**Placeholder this replaces, quoted so the deadline it carried is not lost:**

> Arms, seeds, LoRA config, kill lines, and the callback development slice (drawn from
> `live_simple`, disjoint from both final scoring sets). **Must be committed before arm 1.**

### 3.1 Which track runs is not a choice made here `[FROZEN]`

§2.6's decision table selects the track from the committed calibration artifact. This
section only says what each track is allowed to do:

| `P_std` (§2.6) | Track | Arms permitted |
|---|---|---|
| `>= 1000` | 3A | **A0** confirmatory; A1–A3 exploratory, each under its own written estimate and owner approval |
| `>= 300` and `< 1000` | 3A-cautious | **A0 only.** No exploratory arm may run on this branch |
| `< 300` | 3B | **B0 only** |

**The cadence comes from frozen §2.6, not from this section.** §2.6 states the cautious
branch as *"1 epoch max, eval callback every ~50 steps"*. An earlier draft of §3.8 set looks
every 25 steps and asserted that this satisfied §2.6; it does not, and a candidate section
does not get to reinterpret a frozen one. **Every arm on every track looks every 50 optimizer
steps** (§3.8), which satisfies §2.6 as written and needs no amendment to frozen text. The
cautious branch is then the same arm with the exploratory arms withdrawn.

**An arm not listed in §3.5 or §3.6 requires an amendment before it runs.** That includes
the KTO arm sketched at roadmap Phase 4, an IPO arm, and any additional seed of an arm
already listed.

### 3.2 The development set `D` `[FROZEN]`

Owner Decision C (#general msg 2244) makes `D` a deterministic **258-item look subset** of
the pinned `live_multiple` category, after removing every exact question collision with a
final scoring set and before any study-2 generation. The parent inputs and the subset rule
are pinned independently:

| | |
|---|---|
| parent questions | `eval/bfcl_data/BFCL_v4_live_multiple.json`, sha256 `fd8ccfad4d911420d0e3341dbe2fff77d1d341da934248b9bb2bda24ab3a10c8`, blob `b7c108f2edc86563fd3e1363e5f6fd961452d142` |
| parent answer key | `eval/bfcl_data/possible_answer/BFCL_v4_live_multiple.json`, sha256 `97e90d59c5bd76c55a2920ce93e5566e9046307d3f558578f085f9d3a56c3084`, blob `b2f4c71547d3a534e2e40231db14943bfa81cc74` |
| parent rows / unique ids | 1,053 / 1,053 in each file; answer-name preflight clean **1,053/1,053** |
| parent sorted-id sha256 | `96d9015b2f01ea9a9a090afa8bd8638d81dccccd07d6632379dfc79a35c213ae` (both files) |
| manifest | `eval/manifests/bfcl_v4_study2.json`, sha256 `542d407d434655487daa3faa0da69666cc5e5fa47c8ff67ab9771acc512fe3a0` |
| subset receipt | `mining/receipts/study2_dev_look_subset.json`, sha256 `5a9510711adee429b8d0b2d7e20b35cb57278d052f39cb19d33f86a46b57b33b`, criterion `study2-dev-look-subset/v1` |
| subset | 258 / 258 unique ids from 1,052 eligible; sorted-id sha256 `a91d8271224d7a50f68c27c0070b114173412c2591ba304ac7a6048506760b64` |

**Exact exclusion and disjointness.** The parent has zero id overlap with either final set and
zero exact question overlap with `simple_python`. It has exactly one canonical-question
collision with `multiple`: `live_multiple_190-84-0` ↔ `multiple_26`. The receipt excludes
that item by the general final-question-collision rule, without editing either pinned BFCL
file, leaving 1,052 eligible items. The selected 258 have zero id and zero exact-question
overlap with both final sets. Every count and identity is re-derived by
`tests/test_dev_slice_preflight.py`; drift is a stop condition.

**Seeded, composition-preserving selection.** Within the 1,052 eligible items, the receipt
buckets by presented-tool count `{2, 3, 4, 5–6, 7+}`, allocates 258 slots proportionally by
exact largest-remainder rounding, and orders ids within a bucket by ascending
`sha256(seed + ":" + id)` with seed `study2-dev-look-subset/v1:20260806`. No RNG or library
shuffle participates. The realized bucket allocation is `55 / 74 / 43 / 67 / 19`; it is a
receipt of the fixed rule, not an adjustable target.

**What `D` measures.** All 1,053 parent items present 2–37 candidate tools and every answer
key row expects exactly one call. `D` therefore exercises selection among tools — the skill
under test — unlike the abandoned `live_simple` candidate, whose items each presented one
tool. The development runner must load **exactly** the receipt's 258 ids, match the same 258
answer rows, and refuse missing, duplicate, excluded, or extra ids.

**Sensitivity cost of the subset.** The 258-item subset keeps the look budget at the prior
plan's 258 generations per look, but it deliberately gives up resolution versus all 1,052
eligible items. At 10% discordance and §3.8's exact two-sided `α = 0.0025`, an observed
258-item comparison with `m = 26` discordances needs at least a 21/5 split (net 16 = 6.20
points) for its recorded McNemar diagnostic to cross that display threshold; the full
eligible set at `m = 105` needs 69/36 (net 33 = 3.14 points). These are illustrative exact
rejection thresholds, not power guarantees, and after Decision A they govern neither kill nor
selection. They document the subset's sensitivity cost and why the retired significance rule
was too wide for tie detection. Actual selection uses §3.9's item rule: gaps of at most three
items tie, while a gap of four or more lets accuracy win outright.

**The whole parent category is spent.** The manifest and subset receipt both label
`live_multiple` as `development_selection_only`. Because its subset selects checkpoints,
**none of its 1,053 items — including the 795 outside the look subset — may ever be reported
as a study-2 endpoint.** `tests/test_dev_slice_preflight.py` enforces both role declarations.
The full parent questions file, not merely the 258 selected rows, is included in Amendment
3's decontamination screen so the mining pool is clean with respect to any development item.

### 3.3 The frozen SFT development baseline `[FROZEN]`

**What adoption freezes is the acquisition rule, not a digest.** An earlier draft said the
baseline's sha256 would be "recorded here on adoption", which is impossible in that order:
producing the baseline is a model call, and a model call needs an estimate and the owner's
explicit approval, which come *after* adoption. The corrected ordering, which is also the
order the gates already require:

1. **Adoption** freezes the rule below and nothing else.
2. Under **separately approved spend**, shipped SFT is scored **once** on `D` — greedy,
   `max_new_tokens = 512`, scored by `eval/bfcl_scoring.py`.
3. Its **per-item outcome rows** are committed as
   `eval/results/study2_dev_baseline_live_multiple_subset.jsonl` together with a completion
   record
   naming the run's digest, and that record is **reviewed** before any arm starts.
4. Only then may arm 1 start.

Every kill-line and selection comparison in §3.7 and §3.9 is paired against **those committed
rows**, never against a baseline re-derived at comparison time. The trainer reads the digest
from the reviewed completion record; if the file is missing, or its digest does not match,
**the arm refuses to start** — the same refuse-never-reclassify shape as §2.11. The baseline
is scored once and never re-scored to "refresh" it: a second baseline would silently move
every kill line and every selection that referenced the first.

**The completion record states `n_base` explicitly** — the raw correct count out of 258 and
the accuracy it implies — rather than leaving it to be re-derived from the outcome rows. Every
kill line and every eligibility test in §3.7 and §3.9 is denominated in items against that one
number, so it is recorded as a first-class quantity, not as a by-product.

Recording it changes no threshold. `D` is now a `live_multiple` subset rather than the
single-tool `live_simple` candidate, so shipped SFT's accuracy on it is genuinely unknown
until this run happens; **the rules do not care.** §3.7's kill margin and §3.9's eligibility
margin are absolute item counts against `n_base` whatever `n_base` turns out to be (§3.7 rule
4). `n_base` is reported because a reader should be able to see the baseline the thresholds
were applied to — not because seeing it licenses moving them.

This baseline is one candidate × 258 greedy generations and is counted in §3.11.

### 3.4 Configuration — common settings, per-track schedules, and overrides `[FROZEN]`

An earlier draft put every setting in one "fixed across every arm" table, which then
contradicted itself: B0 trains at LR `1e-4` and effective batch 32, not the `5e-6`/16 that
table declared universal. The three tables below say which is which, so nothing has to be
read past.

**(a) Common to every arm, both tracks:**

| | |
|---|---|
| Base | `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659` |
| Init | shipped SFT adapter, revision `b6f4da479f8c6fc044ee8b802a92f47780f970c5`, **trained in place** (`is_trainable=True`) |
| LoRA | `r = 64`, `alpha = 128`, `dropout = 0.05`, targets `["q_proj", "k_proj", "v_proj", "o_proj"]` — identical to study-1 SFT (`train/sft_full.py`). No new modules, no rank change |
| Epochs | 1 |
| Precision | bf16, gradient checkpointing on, `max_length = 2048` |
| Split | target 90/10 train/eval within every non-empty track-specific cell, split seed `42`; exact integer rule and **track-specific key** below |
| Seeds | training seed `42`, split seed `42` |
| Look cadence | every **50** optimizer steps, `L_max = 20` (§3.8) |
| Decoding | **greedy** for every dev look and every final score — no sampling seed enters any measurement |

**(b) Track 3A only (DPO arms A0–A3):**

| | |
|---|---|
| Schedule | per-device batch 2 × grad-accum 8 (**effective 16**), peak LR `5e-6`, cosine, `warmup_ratio = 0.03` |
| Reference | the SFT policy, via the frozen `ref` adapter — asserted, not assumed (below) |
| Split key | the pair's `error_type` — every 3A row **is** a preference pair and carries exactly one; each error-type value is one split cell |

**(c) Track 3B override (B0 only):**

| | |
|---|---|
| Schedule | per-device batch 8 × grad-accum 4 (**effective 32**), peak LR `1e-4`, cosine, `warmup_ratio = 0.03` |
| Objective | LoRA-SFT continuation — no preference objective, so no reference model, and 3A's preference-metric kill lines do not apply (§3.7) |
| Split key | `bucket × stratum` — four cells: `{0/8, 1–7} × {multi, single}` (below) |

**Why B0 cannot use 3A's split key, and what it uses instead.** A B0 row is not a preference
pair: a `0/8` row is a prompt plus its **ground-truth** answer and has eight sample verdicts
rather than one `error_type`, so stratifying it by `error_type` is not merely awkward, it has
no value to read. B0 therefore stratifies on facts every row does have:

> **Key = (`bucket`, `stratum`)** where `bucket ∈ {0/8, 1–7}` (§3.6) and `stratum ∈ {multi,
> single}` (frozen §2.2) — four possible cells, split independently with seed `42`.

`source` (`xlam` / `hermes`) is **reported, not balanced**: it is a provenance label, it is
already strongly correlated with `stratum`, and balancing on it would shrink cells for no
inferential gain. Row counts by `source × bucket × stratum` are reported with the composition.

**Exact rounding, applied to both tracks.** For each non-empty cell — an `error_type` cell in
3A, or a `bucket × stratum` cell in B0 — let `n_cell` be its row count and set
`n_eval = max(1, ceil(0.10 × n_cell))` and `n_train = n_cell - n_eval`. The realized aggregate
train/eval counts and proportions are reported; "90/10" names the target, not a claim that
every small integer cell realizes it exactly.

**Fail-closed on thin cells.** Every non-empty cell must place at least one row on each side
of the split. Therefore `n_cell >= 2` is required for **every 3A and B0 cell**; a thinner cell
**stops the run** with the track and cell named. It is never merged into a neighbour, never
dropped, and never silently sent entirely to train: each of those is a reclassification of
data the split rule could not handle, which is the pattern §2.11 exists to forbid.

Where §3.6 says B0 shares "everything else" with 3A, it means table (a), the development set,
the frozen baseline, the dev-health kill lines, the selection rule, and the analysis — not
table (b).

**One seed per arm. No seed replication.** Recorded as a limitation now rather than
discovered later: differences *between* arms cannot be separated from seed variance. That is
precisely why exactly one arm is confirmatory (§4.1) and the rest are labelled exploratory.

**Library pins, and the reference-model semantics they decide.** TRL's reference model for a
**re-trained** PEFT adapter is version-dependent:

- **TRL 1.8.0** (installed here) copies the pretrained adapter into a frozen `ref` adapter
  and computes reference log-probs from it → the reference is **the SFT policy**.
- **Older TRL** computed reference log-probs with adapters disabled → the reference is **the
  base model**.

These are different experiments. `pyproject.toml` pins only `trl>=0.10`, and the study-1
DPO v2 run's TRL version is **not recorded anywhere in this repository** (searched `docs/`,
`eval/`, `train/`, `eval/results/`). **Study 2 therefore does not claim library-level recipe
identity with ADR-008**, and no wording in the write-up may imply it.

Study 2 pins, and the trainer asserts at start, refusing on mismatch:

| | |
|---|---|
| `trl` | 1.8.0 |
| `peft` | 0.19.1 |
| `transformers` | 5.14.1 |
| `torch` | 2.13.0 |
| `datasets` | 5.0.0 |
| `accelerate` | 1.14.0 |

The trainer additionally asserts that a `ref` adapter exists on the policy — i.e. that the
reference **is** the SFT policy — rather than inferring it from the version string.

The versions above are **this workstation's**, captured 2026-08-06. **No pod image artifact
exists yet**, so "the pod matches" is a *precondition to be established*, not a fact: before
arm 1, the pod's resolved versions are captured into the run manifest and asserted against
this table, and a mismatch is a stop-and-record, never a silent proceed. Writing it as though
a matching image already existed would be the same class of claim — a check described rather
than performed — that this document exists to prevent.

### 3.5 Arms — track 3A `[FROZEN]`

| id | role | `loss_type` | `beta` | other |
|---|---|---|---|---|
| **A0-anchor** | **confirmatory (primary)** | `sigmoid` | 0.1 | study-1 v2's *recorded explicit* settings, unchanged |
| A1-beta30 | exploratory | `sigmoid` | 0.3 | tighter anchor to the reference |
| A2-beta05 | exploratory | `sigmoid` | 0.05 | looser anchor |
| A3-sftmix | exploratory | `["sigmoid", "sft"]`, `loss_weights = [1.0, 1.0]` | 0.1 | adds an SFT term on `chosen` |

**Why A0 is the primary and not the cleverest arm.** Study 2 asks one question: does the
answer change when the task has a selection dimension? Of the four arms, A0 is the only one
that changes nothing else on purpose. A1–A3 ask *different* questions and cannot answer this
one.

**What A0 does not license, stated here so no later sentence can borrow the stronger
version.** A0 matches study-1 v2's **recorded explicit settings** — the constants in
`train/dpo_v2_full.py`. It does **not** match a verified study-1 environment, because none
exists: the DPO reference model is part of the objective, TRL's choice of it changed across
versions (§3.4), and study 1's TRL version is recorded nowhere in this repository. **A0
therefore cannot be described as isolating endpoint-and-data as a controlled causal change
relative to ADR-008**, and the write-up may not say "same recipe", "recipe held fixed", or
"only the endpoint changed". The comparison A0 *does* support is the one this document
registers: shipped SFT versus A0's selected checkpoint, on `multiple`, measured within study 2
under one pinned environment. If a traceable study-1 environment artifact is ever recovered,
the stronger claim becomes available by amendment — not by rewording.

**A3 is the "anchored arm" roadmap 3A.1 asked us to pick and state.** Roadmap 3A.1 offered
DPOP or a DPO + SFT-loss mix on `chosen`; we state the SFT-loss mix, because `rpo_alpha` no
longer exists in TRL 1.8 and the supported equivalent is multi-loss `["sigmoid", "sft"]` with
explicit `loss_weights`. It is the theory-driven arm: ADR-008's failure mode was correct
outputs degrading on held-out prompts, and an SFT term on `chosen` targets exactly that. It
is nonetheless **exploratory**, because a study cannot both fix a recipe and claim its fix as
the primary result.

**Run order: A0 first, alone, to completion or kill.** No exploratory arm runs before A0 has
finished and its result is recorded.

### 3.6 Arm — track 3B `[FROZEN]`

| id | role | method | LR | epochs | effective batch |
|---|---|---|---|---|---|
| **B0-rsft** | **confirmatory** when §2.6 routes to 3B | LoRA-SFT continuation of the shipped adapter | `1e-4` | 1 | 32 (8 × 4) |

Roadmap 3B.2 permitted `1e-4`–`2e-4` and required the value be stated: **`1e-4`**, the
conservative end, because B0 continues an already-converged adapter.

**Data — the 0/8 bucket is the point of this track, not an optional extra.** An earlier draft
built B0 from accepted self-generations alone, which drops exactly the prompts the track
exists for: 3B fires when yield is *low*, and a low-yield pool is one where the model rarely
succeeds. Restored to roadmap 3B.1, stated deterministically so nothing is decided later:

`data/rsft_train.jsonl` = **both** of the following, one row per prompt, no prompt appearing
twice:

| source | rows | completion used |
|---|---|---|
| **0/8 prompts** (`mining_out/sft_bucket.jsonl`) — the model failed all 8 samples | all of them | the pool's **ground-truth** answer |
| **1–7 zone prompts** — at least one sample passed the verifier | all of them | the **verifier-accepted sample with the lowest sample index** |

The 1–7 rows are **included**, not optional: roadmap 3B.1 left that open and required the
composition be stated, and stating it as a rule beats stating it as a count discovered later.
Selection within a prompt is deterministic and quality-blind beyond the verifier's binary
verdict — **no "best of 8" by any score**, which would be selection on an unregistered
metric. Composition (row counts by source and by stratum) is reported before training starts,
and a prompt appearing in both buckets is impossible by construction — 0/8 means no sample
passed.

Everything else is shared with 3A per §3.4(a): same `D`, same frozen baseline, same
**dev-health** kill lines, same selection rule, same analysis. The preference-metric kill
lines (§3.7 rules 2 and 3) do not apply — B0 has no preference objective and no reference
model.

### 3.7 Kill lines — mechanical, in code, per arm `[FROZEN]`

Each is a `TrainerCallback` that stops the arm and writes `kill_report.json` naming the rule,
the look index, the metric values, and the optimizer step. **A human watching is not a
control** (ADR-007's lesson, retained).

1. **Non-finite** — NaN or inf in the loss or any logged metric → stop immediately.
2. **Preference-metric damage** — `eval_rewards/chosen < -0.25` → stop. ADR-007's line,
   unchanged. *(3A only.)*
3. **Easy data** — first-eval `eval_rewards/accuracies >= 0.99` → stop; the pairs are
   trivial and the arm cannot be informative. *(3A only.)*
4. **Dev health — absolute, and stricter than the roadmap's line.** Let `n_base` be the
   number of items the frozen SFT baseline (§3.3) gets right on `D`, and `n_look` the number
   the current policy gets right at this look, both out of 258.

   > **Stop when `n_base - n_look >= 3` at two consecutive looks** (3 of 258 = **1.16
   > points**, so this triggers no later than roadmap 3A.1's "below SFT baseline −1.0 pt on
   > two consecutive evals"),
   > **or immediately at any single look where `n_base - n_look >= 26`** (10.0 points).

   Two consecutive looks for the ordinary rule, because a single look moves on noise; the
   collapse floor at one look, because a collapse should not have to wait for a second.

   **Both thresholds are absolute and do not depend on the baseline's level.** A 3-item loss
   is a 3/258 = 1.16-point loss whether shipped SFT scores 240 or 140 on `D`; a 26-item loss
   is 10.0 points on either. Decision C moved `D` from a single-tool set to a multi-tool one,
   and **no committed artifact measures either candidate's baseline** — study 1's committed
   scores are for `simple_python`, not `live_simple` — so the change is one of structure, not
   of a known difficulty level. **That changes nothing here by design.** Any
   baseline-dependent replacement — a margin scaled to headroom,
   to `n_base`, or to anything else — requires an **exact pre-stated function and an owner
   amendment before arm 1**. It may never be introduced as judgement applied after `n_base`
   is observed, which would be a threshold chosen in sight of the data it judges.

   **This replaced a significance-based trigger, and that was a real defect.** The earlier
   draft stopped only when the damage reached `α_look = 0.0025` on exact McNemar. On n = 258
   that threshold needs a large lopsided discordant split, so an arm could sit several points
   below the baseline indefinitely without ever tripping a rule the roadmap had already set
   at 1.0 point. **A kill line must be an absolute floor**; a significance test is not one,
   because its trigger point moves with the discordance the run happens to produce.
5. **Direction guard** — a look where the policy is *ahead* of baseline stops nothing. It may
   affect the within-arm checkpoint ranking in §3.9, but `D` is not an endpoint (§3.2), and a
   dev win is not a study result.

**Exact McNemar against the frozen baseline is still computed at every look and recorded** —
`b`, `c`, and the p-value against `α_look` (§3.8) — as a **reported diagnostic**. It triggers
no kill, and since Decision A (§3.9) **no McNemar comparison of any kind has authority over
selection either**: the tie set is an absolute item cap. Every **per-look McNemar diagnostic
in §3** is therefore a recorded number that governs neither kill nor selection. This does not
apply to §4's final-analysis McNemar tests, which govern the study's inferential verdicts.

A killed arm keeps every artifact it produced, is reported as killed (§4.7), and is **not**
silently replaced by another arm or another seed.

### 3.8 Repeated looks have a stated multiplicity rule `[FROZEN]`

Looks run every **50 optimizer steps** — the cadence frozen §2.6 states for the cautious
branch, applied to every arm on both tracks so no candidate section ever reinterprets a
frozen one — capped at **`L_max = 20` looks per arm**:

> `cadence = max(50, ceil(total_steps / 20))`
>
> Looks occur at each multiple of `cadence`, **plus a mandatory look at the final optimizer
> step** when that step is not already a look. Therefore `L = ceil(total_steps / cadence)`,
> and **`L >= 1` for any run of at least one step**.

**The final-step look is not a nicety; without it a short arm has no candidate at all.** An
earlier draft wrote `L = min(20, floor(total_steps / 50))`, which returns **zero** looks for
any arm under 50 steps — no dev measurement, no saved look checkpoint, and therefore nothing
selectable under §3.9. A 500-pair mine (29 steps) would have produced an arm that trained
successfully and could not be used.

`L <= 20` holds by construction: when `total_steps > 1000` the cadence stretches to
`ceil(total_steps / 20)`, and `ceil(total_steps / ceil(total_steps / 20)) <= 20` for every
positive integer (verified by exhaustive check over `total_steps` in 1..20,000). The realized
cadence and look count are recorded in the run artifact.

> **`α_look` = 0.05 / `L_max` = 0.0025**, Bonferroni over the **maximum** look count, fixed
> in advance and **not** recomputed from the realized number of looks.

Recomputing `α_look` after seeing how many looks actually happened would be the same defect
one level down from the one Amendment 2 fixed: a threshold that moves with the data it
judges.

**What `α_look` is now for, and what it is not.** **Nothing mechanical depends on it.** Kill
lines are absolute counts (§3.7 rule 4) and, since the owner's Decision A (#general msg 2292),
selection is an absolute item cap with no significance test anywhere in it (§3.9). `α_look`
survives for exactly one purpose: giving the recorded per-look McNemar diagnostic a
multiplicity-aware display threshold, so a reader is not handed twenty bare `p < 0.05`
opportunities to over-read. Bonferroni over `L_max` fixes it before the realized look count is
known. **These p-values are never endpoint inference:** they enter none of §4's families and
are never reported as evidence of a study effect.

### 3.9 Checkpoint selection `[FROZEN]`

A checkpoint is saved at **every look**, including the mandatory final-step look (§3.8), so
the set of selectable checkpoints and the set of dev measurements are the same set. Per arm:

> **Eligible** = a checkpoint whose dev correct-count satisfies `n_ckpt >= n_base - 2` — a
> fixed **non-inferiority margin of 2 items (0.78 points)** against the frozen SFT baseline.
>
> Among eligible checkpoints, the **leader** is the one with the highest dev correct-count.
>
> The **tie set** is every eligible checkpoint within **`K = 3` items** of the leader's
> correct-count on the 258-item subset. Beyond `K`, **accuracy wins outright**.
>
> **Selected** = the checkpoint in the tie set with the most optimizer steps. **If no
> checkpoint is eligible, the arm has no candidate** (§4.7).

**No significance test appears in selection, and that is the point.** An earlier candidate
defined the tie set by exact McNemar at `α_look = 0.0025`. The owner retired that rule and
specified the one above (#general msg 2292), naming the defect in his own rider: it stated an
intent — *"a point apart is noise"* — and then named a mechanism that contradicts it by a
factor of six. Two things were wrong with borrowing that threshold:

- **`α_look` was calibrated for a different question.** It *was originally* set so an arm
  would rarely be killed by chance across up to 20 looks. Nothing about that calibration makes
  it a tie detector — and after Decision A nothing mechanical depends on it at all (§3.8).
- **Failing to reject is not equivalence.** A 6-point gap that misses significance at
  `α = 0.0025` is not a tie; it is a difference the test was underpowered to confirm. Calling
  it a tie is the same error, one level down, that §3.9's eligibility margin already fixed by
  replacing "not significantly worse" with an absolute count.

**And the wide reading was not merely conservative — it was directionally biased against the
candidate.** With everything inside a ~6-point band tying and step count deciding, the rule
collapses to *final-checkpoint-unless-vetoed*: Option B wearing §6-compliant clothing. Against
a peak-then-degrade trajectory — **the only trajectory study 1 observed**, where its committed
per-item rows give 364 / 363 / 359 correct at steps 50 / 100 / 150 — it would systematically
select **past** the peak. Decision C's entire selection benefit would evaporate at the moment
of selection. One trajectory is not a law, which is why this is an argument about the rule's
direction of bias rather than a prediction about study 2's curve.

**Why `K = 3` items.** It is unit-exact — items, not points with rounding ambiguity — and
`3 / 258 = 1.16 points` is **the nearest integral calibration to the rider's "a point
apart"**.

It is also anchored to measured data rather than intuition, with the arithmetic stated
exactly rather than rounded in our favour. Study 1's checkpoint-to-checkpoint spreads,
recomputed from `eval/results/study1_bfcl_simple_generations.jsonl`, were **0.25 to 1.25
points on 400 items**. Carried across to a 258-item subset, those effect sizes are **0.65 to
3.23 item-equivalents**. An observed 258-item gap must be integral, so the 1.25-point endpoint
has no exact realization there: it falls between a **3-item gap (1.16 points), which ties**,
and a **4-item gap (1.55 points), which wins on merit**. The historical range motivates the
location of the boundary but cannot decide which side a future observed count occupies; in
particular, the preregistration makes no claim that every past-like spread must tie.

`K = 5` was offered as an explicitly conservative alternative and is **not** taken: a wider
band re-introduces the step-count bias this rule exists to remove, and the argument for
narrowing does not stop halfway. No standard-error justification is offered for either value —
shipped SFT's accuracy on `D` is unmeasured until §3.3 runs, so any SE quoted for a 258-item
subset today would be computed from an accuracy nobody has observed. Pinned now, before any
curve exists that a choice of `K` could be made to flatter.

**The eligibility margin is strictly tighter than the kill margin, and has to be.** The kill
line fires at a 3-item loss sustained over two looks (§3.7 rule 4); an earlier draft made
eligibility `n_ckpt >= n_base - 3`, which called that same 3-item-loss checkpoint *selectable*.
A checkpoint at the kill boundary would then have been eligible to become the arm's candidate
— the run stops for damage the selection rule was willing to ship. At `n_base - 2` the two
boundaries cannot cross: anything the kill line would stop for is already ineligible. A
last-look first strike may go un-killed (one look, no second to confirm), but it can never be
selected.

This ranks first by the held-out task metric, as WORKING-AGREEMENT §6 requires, while refusing
to let sub-resolution count noise automatically pick an earlier checkpoint. No threshold is
inherited from the look schedule: the cap is `K` items, fixed here, and it neither tightens
nor loosens with an arm's realized length. Not best `eval_loss`. Not best reward margins. Not
any number from `multiple` or `simple_python` — **the final scoring sets are not opened until
the checkpoint is already selected.**

**Why an absolute margin and not a significance test.** An earlier draft made eligibility
"not significantly worse at α = 0.05". That was wrong twice over: failing to reject harm is
not evidence of health, and applying it across up to 20 saved checkpoints tests the same
baseline twenty times at an unadjusted threshold. A fixed margin has neither problem — it
makes no test, so it has no multiplicity, and it states in advance exactly how much dev
degradation the study is willing to carry into a candidate.

**Owner decision: Option C, 2026-08-06 (#general msg 2244).** The earlier `live_simple`
proposal could gate single-tool health but was blind to the multi-tool ranking skill under
test. That blindness did **not** prove it would systematically select the least-trained
checkpoint — the relationship between its accuracy and optimizer step was unmeasured — but
it left the hypothesis unobserved during selection. The owner therefore selected the
skill-visible `live_multiple` development design in §3.2 while no study-2 generation existed.
The choice amends frozen counts and inputs through Amendment 3; it does not amend §6 or weaken
the rule against selecting on training/preference metrics.

### 3.10 Before arm 1 `[FROZEN]`

Every item is a precondition; any failure is a stop, not a warning.

- **Fixtures dry-run:** 5 steps, `learning_rate = 0.0`, throwaway output directory, asserting
  parameter-hash equality before and after (WORKING-AGREEMENT §5). The directory is deleted
  and no dry-run artifact enters evidence or selection.
- **Library pins assert clean** against the pod's resolved versions, and the `ref` adapter is
  present on 3A arms (§3.4).
- **Dev baseline run, committed, reviewed, and digest-matched** (§3.3) — a separately
  approved spend that completes *before* arm 1, not alongside it.
- **`D`'s pinned counts and digests re-verified** from the files (§3.2).
- **The WORKING-AGREEMENT §6 selection question answered by the owner** — Option C,
  #general msg 2244 (§3.9). Before any generation, the `live_multiple` questions and answer
  key, the overlap exclusion, the seeded dev-subset receipt, the amended manifest, the
  re-screened decontamination artifact and weights, and Amendment 3 must all be committed,
  reviewed, and public. The miner consumes the re-screened pool; it may not mine the
  superseded artifact.
- **§3 and §4 adopted by the owner and public** before any arm starts.

### 3.11 Spend `[FROZEN]`

**Adoption of this section authorizes no spend.** Every stage below needs its own written
estimate agreed by the agents and explicitly approved by the owner (WORKING-AGREEMENT §3).
What is stated here is the *countable work*, so the estimate is arithmetic rather than a
guess:

| stage | countable work |
|---|---|
| dev baseline (§3.3) | 258 greedy generations × 1 candidate — **owner-approved spend, and it must land and be reviewed before arm 1** |
| one 3A arm | 1 epoch of DPO at effective batch 16, plus `L` dev looks × 258 greedy generations (`L` per §3.8, worked below) |
| one 3B arm | 1 epoch of LoRA-SFT at effective batch 32, plus the same dev-look budget |
| final scoring (§4) | per candidate: 200 (`multiple`) + 400 (`simple_python`) greedy generations |

`L = ceil(total_steps / cadence)` with `cadence = max(50, ceil(total_steps / 20))` (§3.8).
For a 3A arm, the exact split rule in §3.4 gives
`train_rows = sum(n_cell - max(1, ceil(0.10 × n_cell)))` over the realized `error_type`
cells, then `total_steps = ceil(train_rows / 16)`. Before the cell composition exists,
`train_rows <= floor(0.90 × mined_pairs)`, so the table gives conservative upper bounds,
not claimed realized counts:

| mined pairs | upper-bound optimizer steps | cadence | upper-bound `L` | upper-bound dev generations |
|---:|---:|---:|---:|---:|
| 500 | 29 | 50 | 1 | 258 |
| 1,000 | 57 | 50 | 2 | 516 |
| 2,500 | 141 | 50 | 3 | 774 |
| 5,000 | 282 | 50 | 6 | 1,548 |
| 12,000 | 675 | 50 | 14 | 3,612 |
| 16,890 | 951 | 50 | **20 (capped)** | **5,160** |

The conservative upper bound first reaches the **5,160-dev-generation ceiling per arm** at
16,890 mined pairs; the realized cellwise split can produce fewer train rows, steps, and
looks. Every positive-step arm still has at least one final-step look, so no realized yield
produces an arm without a selectable checkpoint.

**Flagged before anyone is surprised by it:** the dev looks, not the training, dominate an
arm's inference cost. The cadence is the knob if the written estimate comes back higher than
the owner wants to spend — changed by amendment to §3.8 **before** the arm runs, never after
seeing a curve.

## 4. Final analysis `[FROZEN — owner-adopted 2026-08-06; exact-content review cleared 2026-08-07]`

> **Adoption line.** Adopted by the owner on 2026-08-06 (#general msg 2297), in the same
> instruction that adopted §3: *"Adopt §3 and §4, publish Amendment 3, unblock the miner, and
> send me the pilot estimate when it exists."*
>
> **No study-2 model output, evaluation result, probe score, mined pair, or yield number had
> been observed at adoption. No model had been run and model spend was $0.**

This matters more here than anywhere else in the document: **every family, margin, band, and
decision rule below was fixed before a single number existed to fit them to.** The text of the
original placeholder is retained unaltered as §4.0, because its power note is already the
pre-registered MDE statement and nothing here supersedes it.

### 4.0 Carried forward unaltered from the `[PENDING]` placeholder

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

### 4.1 Three families, fixed before the data exist `[FROZEN]`

**Confirmatory — exactly one contrast.** Shipped SFT versus the selected checkpoint of
**A0** (track 3A) or **B0** (track 3B), on `multiple`, n = 200. One contrast, so no
multiplicity adjustment applies to it, and **no other contrast may be reported as "the
result"**.

**Secondary — exactly one tested contrast:** the structural stratum
`len(function) in {3, 4}`, n = 121 (A1.5), with both id-set digests re-verified at analysis
time. A family of one takes no multiplicity adjustment; if a future amendment adds a second
secondary **test**, the family becomes Holm-corrected and that amendment says so.

**Two guardrails sit outside every family, because neither is a test:** the `simple_python`
retention band (§4.1a) and the MMLU capability band (§4.5). Both compare a measured number to
a line fixed in advance and gate the shipping recommendation. Holm-correcting them would be a
category error — a family is a set of tests, and a band is not one.

An earlier draft had retention inside a Holm family while deciding it from an unadjusted
interval, which is two procedures wearing one name. It is now one procedure: a band.

**Exploratory — one contrast per exploratory arm actually launched:** the arm's selected
checkpoint versus shipped SFT, on `multiple`, n = 200 — the same endpoint as the confirmatory
contrast. **Family size = the number of exploratory arms launched (0 to 3), recorded before
the analysis runs**, and Holm-corrected over exactly that count. Always reported with both raw
and adjusted p-values. **An exploratory arm cannot be promoted to confirmatory after the
fact.** Designating the primary before any data exist is the only thing that makes that
promotion detectable, and this is the sentence that forbids it.

### 4.1a Retention is a band on the measured difference `[FROZEN]`

"No significant loss" is not evidence of retention — it is the absence of evidence of loss,
which a small or noisy sample buys for free. Retention is therefore a **guardrail on the
observed paired difference**, with no test and no p-value:

> **`simple_python` retention fails if the candidate's net loss against shipped SFT on the
> same 400 items is 8 or more items** — i.e. at least 2.0 percentage points. Otherwise it
> passes. The exact per-item counts `b` and `c`, the net difference, and the 95% Tango
> interval are reported either way — the interval informs the reader, it does not decide the
> band.

**Operating characteristics, computed rather than asserted.** An earlier draft claimed a
2.0-point margin "is not a margin that noise alone can breach". **That was false**, and the
arithmetic says so. Under no true difference, the discordant split is symmetric, so
`P(net loss >= 8 items)` depends entirely on how many discordant items there are:

| discordant items `m` | P(retention fails by chance alone) |
|---:|---:|
| 10 | 0.011 |
| 16 | 0.038 |
| 20 | 0.058 |
| 30 | 0.100 |
| 40 | 0.134 |
| 60 | 0.183 |

(Exact, `b ~ Binomial(m, 0.5)`; reproduce with the closed form in §4.3's reporting code.)

Two things follow, and both are stated rather than hidden. **The band is chosen at 2.0 points
with its false-failure rate known**, not assumed to be zero: at study 1's observed discordance
for SFT-vs-DPO contrasts (`m = 16` for the widest, ADR-008's recovered per-item rows) it is
~4%, rising to ~18% if study 2's candidates prove far more discordant. And **the direction of
that error is the safe one**: a false retention failure withholds a shipping recommendation
and forces the mixed row of §4.6 — it can never manufacture a positive claim. The observed `m`
is reported alongside the verdict, so a reader can place the result on this grid exactly as
A1.3's grid places the primary contrast.

Why 2.0 points (8 items) and not 1.5 or 2.5, on the same `>=` convention throughout:

| band | chance failure at `m = 16` | verdict on study 1's worst checkpoint (−10 items) |
|---|---:|---|
| 1.5 pts (≥6 items) | 10.5% | fails |
| **2.0 pts (≥8 items)** | **3.8%** | **fails, with 2 items to spare** |
| 2.5 pts (≥10 items) | 1.1% | fails exactly at the boundary |

1.5 points is too noisy to gate anything. 2.5 points catches ADR-008's worst checkpoint only
by landing precisely on the line, which leaves a rule with no margin against the one bad case
we have actually observed. 2.0 points fails it with room and holds chance failure near 4% in
the discordance regime study 1 exhibited. Fixed before any candidate exists, with its
justification in the same paragraph, so it cannot be re-derived later to fit a result.

### 4.2 Method `[FROZEN — carried from A1.4]`

Exact two-sided McNemar on discordant pairs, plus **Tango's score interval** for the paired
difference of proportions, 95%, reported regardless of direction or significance. Marginal
binomial CIs are reported and decide nothing.

**Implemented, with its provenance stated rather than implied.** `eval/paired_analysis.py`
computes exact McNemar, Holm adjustment, and — since commit `92346e6` — Tango's score
interval, exposed for every contrast and in the CLI. A1.4 named the interval before the code
existed; the code has now caught up to the document rather than the document being trimmed to
the code.

The interval is validated two ways, because either alone leaves a hole: **conformance**
against `PropCIs` 0.3-0 (a GPL-licensed third-party implementation citing Tango 1998, used as
a checksum-pinned black-box oracle — no code copied into this Apache-2.0 repository, and the
tests run offline with pinned values), and **structural** checks that the endpoints solve the
score equation, bracket the point estimate, mirror under swapping `b`/`c`, and honour the
documented degeneracies. **The oracle vectors are arbitrary synthetic counts, not study
data**, and neither R nor PropCIs is a runtime or CI dependency. Tango (1998) itself is
paywalled and no accessible worked table was found, so nothing here claims to reproduce a
primary-source table.

### 4.3 Discordance is reported, not just the p-value `[FROZEN]`

Every contrast reports `b`, `c`, the discordant total, and **which row of A1.3's grid the
observed discordance lands in** — i.e. what the study could and could not have detected at
that discordance. A null is reported with its detectable effect attached, so it can never be
excused as underpowered after the fact or overread as evidence of equivalence.

### 4.4 Callback looks are not results `[FROZEN]`

Every dev-set number from §3.7–§3.9 is a selection or stopping diagnostic, labelled as such,
and enters none of the three families. `D` is never reported as an endpoint (§3.2).

### 4.5 MMLU capability band — a shipping guardrail, not a test `[FROZEN]`

**Not a member of any Holm family (§4.1).** There is no hypothesis here and no p-value: a
measured number is compared to a fixed line, and the line decides whether a candidate may be
recommended for shipping. Study 1's protocol, unchanged: 14,042 items, 5-shot, next-token
log-prob argmax over `' A'/' B'/' C'/' D'`.

> **A candidate whose MMLU is more than 2.0 points below the shipped SFT model is a
> capability regression: it may not be recommended for shipping, whatever its BFCL result.**

The band is set against **shipped SFT**, not base, because the study-2 question is what DPO
adds on top of SFT — study 1 already measured and disclosed the SFT specialization cost
(0.659 vs base 0.683, −2.4 points), and roadmap 5.2 forbids retroactively inventing a tighter
band than the accepted study-1 result. 2.0 points is far looser than DPO v2's observed
0.1-point effect and tighter than the SFT cost, and is fixed here before any candidate
exists.

**When it runs:** MMLU is run **only for a candidate that clears the primary contrast**. At
~90 minutes per candidate it is a shipping check, not a research measurement. If no candidate
clears, MMLU does not run — a stated design consequence, not a missing result.

### 4.6 What each outcome means, fixed before the data `[FROZEN]`

| primary contrast | retention (§4.1a band) | MMLU (§4.5 band) | outcome |
|---|---|---|---|
| significant for the candidate | **passes** (net loss ≤ 7 items) | within band | **Positive result.** Recommended to the owner for shipping; publication remains an owner action |
| significant for the candidate | **fails** | measured either way | **Mixed.** Reported as a trade with both numbers, never as a win; the shipping decision returns to the owner |
| significant for the candidate | passes | outside band | **Blocked on capability.** Not recommended for shipping whatever the BFCL result (§4.5); reported in full |
| significant against the candidate | reported | not run | **Negative result.** Ship SFT; ADR in the ADR-006 / ADR-008 line |
| not significant | reported | not run | **No detectable difference** at the pre-stated MDE, reported with its A1.3 row and its Tango interval. A result, not a failure |

Retention and the stratified contrast are computed and reported in **every** row, including
the rows where they change nothing. A secondary number that only appears when it is
convenient is not a secondary analysis.

### 4.7 Killed arms and empty families `[FROZEN]`

If A0 (or B0) is killed under §3.7, or has no qualifying checkpoint under §3.9, **the
confirmatory family is empty.** The study then reports the kill — rule, look index, metric
values, step — together with the yield-gate outcome and the null.

**An exploratory arm is not promoted to fill the gap, and a second seed is not run to obtain
a survivor.** Either would convert a pre-registered stop into a search, which is the exact
failure this document exists to prevent.

### 4.8 Completeness of the report `[FROZEN]`

- Every arm launched is reported, killed or not.
- Every contrast computed is listed with raw and adjusted p-values, including unfavourable
  ones.
- Yield artifacts, kill reports, `trainer_state.json`, generations, and per-item outcome rows
  are committed, so a reader re-runs the analysis instead of trusting the table.
- Any number this document cannot trace to a committed artifact does not appear in the
  write-up, the README, the model card, or the résumé.

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
| 3 | 2026-08-05 | §2 | Mining section frozen: strata, allocation, sampling, exclusion criterion, artifact-derived weights, yield gate arithmetic | Owner, #general msg 2181 |
| 4 | 2026-08-06 | §2.5, §2.9 | **Amendment 3:** add `live_multiple` to the screen and replace the mining artifact/weights before mining | Owner selected Decision C (#general msg 2244) and adopted the reviewed content (#general msg 2297); publication to the public remote still pending |
| 5 | 2026-08-07 | §3.1, §3.4–§3.5, §3.7–§3.11, §4.1, §4.5 | **Amendment 4:** add the exploratory KTO arm `A4-kto`, trained on the materialized pairs only; widen §4.1's exploratory Holm family from `0 to 3` to `0 to 4` | Owner adopted (#general msg 2625) the reviewed content at `d28e9e1`; publication to the public remote pending |
| 6 | 2026-08-07 | §3.1, §3.4–§3.5, §3.7–§3.11, §4.1, §4.5 | **Amendment 5:** add the exploratory wide KTO arm `A5-kto-wide` over pairs plus the 8-of-8 and 0-of-8 prompts, activated only when `A0_planned_optimizer_steps < 250`; exploratory Holm family becomes **4 when it does not activate, 5 when it does** | Owner adopted Option 1 with the family-size rider (#general msg 2655) on reviewed candidate `a4b1b39`; publication pending |

**Note on this table's first column.** It is a chronological **record** number, not an
amendment number, and the two do not line up. Records 1 and 2 carry numbered Amendments 1 and
2; **record 3 is the §2 freeze, which is a freeze rather than a numbered amendment**; and
numbered **Amendment 3 appears in record 4**, and numbered **Amendment 4 appears in record
5**. A citation of "Amendment 3" therefore means the body section of that name, found at
record 4; "Amendment 4" means the body section of that name, found at record 5; and
"Amendment 5" means the body section of that name, found at record 6. The columns are left as they are rather than
renumbered, because records already cited elsewhere must keep the numbers they were cited by.

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

---

### Amendment 3 — skill-visible development set and re-screened mining pool (2026-08-06) `[ADOPTED by the owner 2026-08-06, #general msg 2297 — publication pending]`

**Owner authorization.** The owner selected Decision C in #general msg 2244: pin a seeded
`live_multiple` look subset, make checkpoint selection rank by its held-out task accuracy,
permanently disqualify the parent category from endpoint reporting, and re-screen the mining
pool before mining. The owner authorized this design and its private-backup work; **this exact text and artifact package were reviewed and then adopted by the owner in
#general msg 2297. Publication to the public remote remains an owner action (WORKING-AGREEMENT
§7) and had not happened when this line was written.**

**Status of data at authorization and drafting: no study-2 model output, development score,
probe score, mined pair, or yield number had been observed. No model had been run and model
spend remained $0.** The counts below are deterministic CPU measurements of pinned input
files and the decontamination predicate, traceable to the committed receipts. They are not
model results and authorize no model call, mining run, arm, or spend.

**Superseded §2.5 text, quoted verbatim and retained above:**

> **Derived weights** (`mining/receipts/sft_dedup_v2_decontamination.json`, sha256
> `fb7a0200dbeeabb831006eeb800a23d3c92d89a468666c61b098ca1277231906`, criterion
> `bfcl-pool-decontamination/v1`):
>
> **`(n_multi, n_single, N)` = (8,173, 2,997, 11,170)**
>
> Stored as exact integers. The multi share
> 8173/11170 = 73.169% is **derived at display time and
> never stored**, so `P_std` is computed from exact ratios.
>
> 12143 cleaned source - 5 prompt-ineligible - 56 target-structural exclusions = 12082 screen inputs; of those, 912 were dropped
> ({'fn_name': 911, 'ngram_overlap': 1}) leaving 11,170 survivors. §1's 12,160-row figures remain
> non-comparable: different source revision, different denominator.

**Superseded §2.9 text, quoted verbatim and retained above:**

> Criterion **`bfcl-pool-decontamination/v1`**, whose predicate pins the 13-gram user-text
> normalization rule, the presented-function-name collision rule, all four manifest
> categories, **the cascade order**, and fail-closed parsing.
>
> - **the SHA-256 of each of the four screened `role=questions` files** from the pinned
>   manifest, as `(category, local_path, sha256)` tuples — the screened question *bytes*, not
>   the manifest file and not the answer keys — plus the manifest's own hash. A future BFCL
>   re-pin then detectably invalidates this artifact instead of silently coexisting with it

The criterion's predicate, cascade order, eligibility rules, and 12,082-row screen-input
population do **not** change. Amendment 3 replaces only the pinned manifest inputs, the
resulting artifact, and the artifact-derived weights, through A3.1–A3.5.

#### A3.1 `live_multiple` is a pinned development input

The manifest now adds `live_multiple/questions` and `live_multiple/answer_key` at upstream
revision `9d8416a96d1d69975493f1b6d60ff07d12a1726a`:

| role | rows / unique ids | sha256 | git blob |
|---|---:|---|---|
| questions | 1,053 / 1,053 | `fd8ccfad4d911420d0e3341dbe2fff77d1d341da934248b9bb2bda24ab3a10c8` | `b7c108f2edc86563fd3e1363e5f6fd961452d142` |
| answer key | 1,053 / 1,053 | `97e90d59c5bd76c55a2920ce93e5566e9046307d3f558578f085f9d3a56c3084` | `b2f4c71547d3a534e2e40231db14943bfa81cc74` |

Both files have sorted-id sha256
`96d9015b2f01ea9a9a090afa8bd8638d81dccccd07d6632379dfc79a35c213ae`, and the standing
answer-name preflight passes **1,053/1,053** rows. The amended manifest sha256 is
`542d407d434655487daa3faa0da69666cc5e5fa47c8ff67ab9771acc512fe3a0`. This extends A2.3's
standing manifest check; it does not change the function-name scoring rule.

#### A3.2 The five-file screen replaces the four-file artifact

The operative decontamination receipt becomes
`mining/receipts/sft_dedup_v2_decontamination_with_live_multiple.json`, sha256
`3daaffa85a2097468f53845d1cddf996a0e68a3605916e26918891c2972732b3`. It binds the amended
manifest and all **five** `role=questions` files, including the full 1,053-row
`live_multiple` parent. Re-running the unchanged criterion gives:

| | frozen artifact | Amendment 3 artifact | change |
|---|---:|---:|---:|
| screen inputs | 12,082 | 12,082 | 0 |
| dropped — function name | 911 | 1,010 | +99 |
| dropped — 13-gram overlap | 1 | 1 | 0 |
| dropped — total | 912 | 1,011 | +99 |
| surviving multi | 8,173 | 8,081 | −92 |
| surviving single | 2,997 | 2,990 | −7 |
| surviving total | 11,170 | 11,071 | −99 |

The amended post-screen id-set sha256 is
`69d381413f8095d483b35c9bcd77e83bd6f72771edc9b4f192510f8e7392e5e3`. The original
`mining/receipts/sft_dedup_v2_decontamination.json` remains untouched as the historical
artifact that froze §2; it is now **superseded and may not feed study-2 mining**.

#### A3.3 §2.5's operative weights are replaced before mining

Applying §2.5's unchanged artifact-derived rule to the Amendment 3 artifact gives:

> **`(n_multi, n_single, N)` = `(8,081, 2,990, 11,071)`**
>
> `w_multi = 8081/11071 = 72.993%` and `w_single = 2990/11071 = 27.007%`, displayed only;
> `P_std` continues to use the exact integer ratios.

The previous multi share was `8173/11170 = 73.169%`; the amended share is lower by 0.177
percentage points (0.18 points at the precision used in the owner decision). These are
artifact-backed pool-composition measurements, not portfolio targets or model outcomes.

#### A3.4 The development subset removes the final-set collision and spends the parent

`mining/receipts/study2_dev_look_subset.json`, sha256
`5a9510711adee429b8d0b2d7e20b35cb57278d052f39cb19d33f86a46b57b33b`, applies
`study2-dev-look-subset/v1` to the parent. It removes the one exact canonical-question
collision, `live_multiple_190-84-0` ↔ `multiple_26`, then deterministically selects 258 of
1,052 eligible ids under seed `study2-dev-look-subset/v1:20260806`. The selected sorted-id
sha256 is `a91d8271224d7a50f68c27c0070b114173412c2591ba304ac7a6048506760b64`.

The manifest and receipt both label `live_multiple` as `development_selection_only`.
Selection spends the **entire parent category**, not only the chosen rows: no score on any of
its 1,053 items may be reported as a study-2 endpoint. The exclusion exists only in the
receipt; the pinned eval inputs remain read-only and byte-identical to A3.1.

#### A3.5 Sequencing is fail-closed

Amendment 3 must be reviewed, made public, and adopted **before any generation**. Then the
re-screened artifact and exact `(8,081, 2,990, 11,071)` weights must be the inputs to the
miner **before the first prompt is mined**. The miner must verify the amended manifest digest,
the current decontamination-receipt digest, and the post-screen id-set digest; a missing or
superseded input is a refusal, never a fallback to the four-file pool.

Only after that ordering is satisfied may §3's Decision-C development design be adopted.
Every later model or paid-compute stage still requires a written agent-agreed estimate and
the owner's separate explicit approval. This amendment authorizes none.

---

### Amendment 4 — KTO comparison arm (2026-08-07) `[ADOPTED by the owner 2026-08-07, #general msg 2625 — reviewed content d28e9e1; publication pending]`

Frozen §3.1 closes the arm list: *"An arm not listed in §3.5 or §3.6 requires an amendment
before it runs. That includes the KTO arm sketched at roadmap Phase 4."* This amendment adds
that arm. **It authorizes no run:** the arm needs its own written estimate, agent agreement
and explicit owner approval, and it may run only on the branch §3.1 permits.

#### A4.1 KTO trains on the materialized pairs only

Each pair in `mining_out/mined_pairs.jsonl` contributes one **desirable** row (`chosen`) and
one **undesirable** row (`rejected`). The 8-of-8 and 0-of-8 prompts the miner discards are
**not** admitted.

A0-versus-KTO compares *objectives*. Admitting the unpaired prompts would change data volume
**and** composition at once, so a difference could not be attributed to the objective. A
wider design admitting those buckets is a **separate candidate amendment**, not a variant of
this one.

#### A4.2 The arm, and where it is permitted

Adds one row to §3.5:

| id | role | `loss_type` | `beta` | other |
|---|---|---|---|---|
| **A4-kto** | **exploratory** | `kto` | 0.1 | unpaired objective over the converted pair set |

**Permitted only on the `P_std >= 1000` branch.** §3.1 already states the `300–999` branch
runs A0 only and that no exploratory arm may run there; `A4-kto` inherits that unchanged, and
**this amendment widens no branch**. Run order is unchanged: A0 first, alone, to completion
or kill.

**`A4-kto` is exploratory permanently.** It may never be the confirmatory candidate or the
structural secondary, and §4.1 forbids promotion after the fact.

#### A4.3 Split first, then expand; the dataset contract

The 90/10 split runs over **pairs**, under §3.4's exact integer rule, track-specific cell key
and split seed `42` — *then* each pair expands into its two rows. Splitting after expansion
would place a prompt's desirable and undesirable rows on opposite sides of the held-out
boundary and leak it.

> `pair_id = f"{prompt_id}:{chosen_index}:{rejected_index}"`, asserted unique.

| field | value |
|---|---|
| `pair_id` | as above — the linkage key |
| `prompt_id`, `chosen_index`, `rejected_index`, `stratum` | copied unchanged from the source pair |
| `prompt` | exactly the pair's `prompt_messages` list |
| `completion` | exactly `[{"role": "assistant", "content": <chosen or rejected>}]` |
| `label` | `bool` — `true` desirable, `false` undesirable |
| `verifier_version` | exactly `onpolicy_verifier_v1`; any other value fails the conversion |
| `error_type` | `= rejected_reason`. The recognized set is exactly `mining.verifier.REASONS` under `onpolicy_verifier_v1`: `{invalid_json, missing_call, spurious_call, wrong_tool, wrong_args}`. Null or any other value fails |

**Asserted fail-closed before the arm starts:** every `pair_id` appears exactly twice with
opposite labels; no `pair_id` and no `prompt_id` appears in both splits; each split holds at
least 2 pairs. The input file, the split receipt and both converted outputs are digested.

#### A4.4 Row order is part of the objective

TRL 1.8's KTO estimates its KL term by rotating completions within the actual batch and
requires `train_sampling_strategy='sequential'`.

> For an ordered split of `N` pairs `p[0..N-1]`, actual batch `i` =
> `[desirable(p_i), undesirable(p_(i+1 mod N))]`. Every batch is balanced, carries two
> distinct `pair_id`s and `prompt_id`s, and leaves no singleton remainder. Requires `N >= 2`.
> **Distributed `world_size = 1`** with one selected CUDA device, asserted at start; the
> physical GPU index is not pinned, the process topology is.

#### A4.5 Batch size, in the correct unit

§3.5's "effective batch 16" is a **pair** unit and does not transfer to unpaired rows.

> Actual batch **2** × gradient accumulation **16** = **32 unpaired rows = 16 source-pair
> equivalents per optimizer step**, one epoch. `total_steps = ceil(train_rows / 32)` where
> `train_rows = 2 × train_pairs`. *Equivalents, not exactly 16 distinct pairs: the cyclic
> cross-prompt ordering means an accumulation window need not contain 16 unique pairs.*

#### A4.6 The rest of the configuration

Inherits §3.4(a) — base `meta-llama/Llama-3.1-8B-Instruct` revision
`0e9e39f249a16976918f6564b8830bc894c89659`, init from the shipped SFT adapter revision
`b6f4da479f8c6fc044ee8b802a92f47780f970c5`, LoRA `r = 64` / `alpha = 128` / `dropout = 0.05`
on the same four targets, bf16, gradient checkpointing on, `max_length = 2048` — and §3.4's
pinned library table, asserted for exact equality at start.

| | |
|---|---|
| `beta` | 0.1 |
| `desirable_weight` / `undesirable_weight` | 1.0 / 1.0 |
| LR · schedule · warmup | `5e-6` · cosine · `warmup_ratio = 0.03` |
| epochs · batch | 1 epoch; per-device 2, grad-accum 16, eval batch 2 |
| `disable_dropout` | `True` — the adapter config retains `dropout = 0.05`; TRL disables the dropout modules for this run |
| `precompute_ref_log_probs` | `False` |
| `sync_ref_model` | `False` |
| sampling | `train_sampling_strategy='sequential'` |
| seeds | training 42, split 42 |
| reference model | frozen `ref` adapter copied from the shipped SFT adapter; parameter-hash equality asserted at step 0 |

**Both weights are 1.0 because A4.1's conversion is exactly balanced**, and 1/1 is the
principled setting for that regime.

**Environment scope, since the two pinned tables in this repository are easy to
confuse.** §3.4's library table governs **training arms** and is what this arm asserts
against. `requirements-probe.txt` scopes itself to the **§0 qualification probe** and does
not supersede it — its deliberately-unpinned-Torch policy applies to §0 inference, not to §3.
The mining pilot's measured `torch 2.8.0+cu128` is a **mining** runtime and is not an arm
pin. **Changing any arm pin requires a separate amendment to §3.4**, not a sentence here.

**Matching `beta` removes one difference; it does not make the objective the only
difference.** KTO's unpaired loss and its in-batch KL construction differ mechanically from
DPO's paired loss even on identical source pairs. The contrast is objective-package versus
objective-package, and the write-up may not describe it as an isolated change of loss
function.

#### A4.7 Kill lines, inherited selectively

§3.7 rules 1, 2, 4 and 5 apply unchanged. **Rule 3 cannot:** TRL's KTO trainer emits no
`eval_rewards/accuracies`.

> **Rule 3 is replaced by `eval_pair_reward_accuracy`.** Per-row reward is
> `beta * (policy_completion_logp - ref_completion_logp)`. A `pair_id` in the held-out split
> counts toward the numerator iff `reward(desirable) > reward(undesirable)` **strictly**;
> **ties count false**. The rate is `numerator / denominator`, both committed with the
> per-pair rows. **First-eval `>= 0.99` stops the arm**, the same threshold rule 3 uses.

A missing or renamed metric fails before training or eval begins, never silently disabling a
kill line. The kill report records numerator, denominator, rate, threshold, split digest,
look index and optimizer step.

#### A4.8 Scoring and analysis

Scored only after every launched arm's checkpoint is selected, on `multiple` (n = 200) and
`simple_python` (n = 400). **Final sets may not influence launch or checkpoint selection.**
The `multiple` contrast versus shipped SFT joins the exploratory Holm family; `simple_python`
retention stays a band outside any test family (§4.1a).

**MMLU is excluded.** §4.5 runs MMLU *"only for a candidate that clears the primary
contrast"*, and `A4-kto` is permanently exploratory, so it can never clear it. Including MMLU
would require amending §4.5.

#### A4.9 Exploratory family size

§4.1 currently reads:

> *"Family size = the number of exploratory arms launched (0 to 3), recorded before the
> analysis runs"*

**Superseded, by reference and without editing that text, to:** *family size = the number of
exploratory arms launched, **0 to 4**, recorded before the analysis runs*, Holm-corrected
over exactly that count.

**The price, accepted by the owner (#general msg 2625):** a fourth possible exploratory arm
widens the correction for A1–A3, which have nothing to do with KTO. It is accepted because
exploratory contrasts can never be promoted to the result, so what is lost is sensitivity in
reporting which exploratory contrasts look individually notable — **not any part of the
confirmatory verdict**, which §4.1 keeps as a family of one taking no adjustment.

#### A4.10 Timing

**This amendment's own stricter rule: it is adopted before the calibration artifact and its
yield are observed.** Current §4.1 does not require this — it defines family size from the
arms actually launched, recorded before the analysis — but a family size settled before any
calibration yield exists cannot be argued about afterwards.

#### A4.11 Spend

Adds to §3.11: *one KTO arm — 1 epoch of KTO at 32 unpaired rows per optimizer step over the
converted pair set, plus `L` dev looks × 258 greedy generations (`L` per §3.8).*

**This amendment authorizes no spend.**

---

### Amendment 5 — wide KTO arm (2026-08-07) `[ADOPTED by the owner 2026-08-07, #general msg 2655 — Option 1 with the family-size rider, reviewed candidate a4b1b39; publication pending]`

Adds a second exploratory KTO arm that admits the prompts the pair miner discards.
**Amendment 4's `A4-kto` trains on materialized pairs only, so that A0-versus-KTO compares
objectives; this arm deliberately does the opposite.** They answer different questions and
both may exist. **This amendment authorizes no run and no spend.**

#### A5.1 The arm

| id | role | `loss_type` | `beta` | other |
|---|---|---|---|---|
| **A5-kto-wide** | **exploratory** | `kto` | 0.1 | unpaired objective over pairs **plus** the 8-of-8 and 0-of-8 prompts |

**Exploratory permanently** — never the confirmatory candidate or the structural secondary,
and never promoted after the fact (§4.1). **Permitted only on the `P_std >= 1000` branch**,
inheriting §3.1 unchanged. **Runs only after A0 completes *or is killed*, and may never
replace A0** whichever way A0 ends. Requires its own written estimate, agent agreement and
explicit owner approval.

#### A5.2 Which rows enter

**One row per admitted prompt — the lowest-index generation of the qualifying kind — not
eight.** Eight near-duplicate completions of one input inflate the set without adding
information and would let a handful of prompts dominate the gradient.

| source | rows | label |
|---|---|---|
| pair prompts (1–7 of 8) | 2, as Amendment 4's conversion | one `true`, one `false` |
| 8-of-8 prompts | 1, generation index 0 | `true` |
| 0-of-8 prompts | 1, generation index 0 | `false` |

**Both solo classes are read from `mining_out/ledger.jsonl` and only from there.** 8-of-8
prompts appear in no other artifact. **`sft_bucket.jsonl` cannot supply the 0-of-8
completions:** its schema is `{prompt_id, prompt_messages, stratum, target}`, it stores **no
generations**, and `target` is the **ground-truth answer** — using it as an undesirable
completion would train the model to avoid the correct answer.

**Reconciliation asserted before conversion, fail-closed** (`mining_summary.json` holds
counts, not IDs): ledger-derived mixed IDs **equal** `mined_pairs.jsonl`'s; ledger-derived
0-of-8 IDs **equal** `sft_bucket.jsonl`'s; 8-of-8 is verified by ledger count **equal to**
`histogram["8"]`; every set size equals its summary count. All four digests recorded.

**Solo row schema:** `pair_id = f"{prompt_id}:solo"`, asserted disjoint from A4's
`prompt_id:chosen_index:rejected_index`; plus `source_index` (0), `bucket`
(`all_correct` / `zero_correct`), `accepted_count`, `prompt` (the ledger record's
`prompt_messages`), `completion`, `label`, `stratum`, `verifier_version` asserted
`onpolicy_verifier_v1`. **Amendment 4's "every `pair_id` appears exactly twice" is superseded
for this arm:** solo `pair_id`s appear exactly once; pair-derived ones still appear twice.

#### A5.3 Weights are this arm's own, not Amendment 4's

**A4.3's `1.0 / 1.0` is superseded for this arm and only this arm.** A4's equal weights were
justified by a dataset **balanced by construction**; this arm runs near **2.8 : 1** at pilot
composition, where equal weights are the wrong setting by KTO's own guidance. **A4-kto keeps
`1.0 / 1.0`.**

> Computed from the realized **post-parity training** split, before the arm runs:
> `desirable_weight = 1.0`; `undesirable_weight = n_desirable_train / n_undesirable_train`
> **as an exact rational**, converted faithfully to float and passed in that form. The run
> artifact records the two counts, the rational and the float. **A rounded display value may
> be reported but never governs training. Fails if either count is zero.**

#### A5.4 Split, parity, and batch order

**Split over prompts at 90/10** under §3.4's exact integer rule and split seed `42`, **then**
expand — no prompt's rows may straddle the held-out boundary. **§3.4's cell key does not
transfer**, because solo prompts have no `rejected_reason`; the key is
`(pair, stratum, rejected_reason)` · `(all_correct, stratum)` · `(zero_correct, stratum)`.
**A cell that cannot yield at least one train and one eval prompt fails the conversion.**

**Parity is resolved before ordering and before the weights are computed.** If a split's
expanded row count is odd, exclude the **lexicographically last eligible solo row** whose
cell remains non-empty, recording `pair_id`, `prompt_id`, cell and reason. **A pair-derived
row is never dropped**; if no eligible solo row exists, the conversion fails.

**Batch order is an exact matching, not a retry loop.** Sort all rows by `(pair_id, label)`
ascending. **Set `minority = D` and `majority = U` when `|D| <= |U|`, otherwise the
reverse** — this makes the equal-size case explicit and deterministic. Walk the minority in
sorted order; match each to the **lowest-sorted unconsumed majority row with a different
`prompt_id`**, **each majority row consumed at most once**. **If any minority row cannot be
matched, the conversion fails** rather than emitting a same-prompt batch that would corrupt
the KL estimate. The even same-label surplus follows in exact sorted order under the same
rule. Asserted: actual batch 2, eval batch 2, `world_size = 1`, **no singleton batch reaches
the KL computation**. The realized per-window label ratio is written to the run artifact.

#### A5.5 Activation — threshold only

> **Activated iff, from the committed calibration artifact alone,**
> `A0_planned_optimizer_steps < 250`, where
> `A0_planned_optimizer_steps = ceil(A0_train_pairs / 16)`.

**This arm may not be launched because another arm's result was disappointing.** Launching in
response to a final-set outcome would require opening the final sets before a launch
decision, which §4 forbids and which would make every downstream contrast outcome-adaptive.

**This is a pre-registered short-run operational threshold, not a power calculation.** Below
250 steps A0 gets at most five 50-step looks (§3.8) — a thin trajectory to select a checkpoint
from. **Step count alone cannot establish statistical underpowering, and this amendment does
not claim it does.**

#### A5.6 Family size is conditional on activation, not on adoption

> **`family_size = 4` when A5-kto-wide does not activate; `family_size = 5` when it does**,
> fixed from the committed calibration artifact **before any arm runs** and recorded there.

**This clause is the reason Option 1 was chosen over Option 2** (owner, #general msg 2655). If
adoption alone set the family to 5, Option 1 would pay Option 2's full statistical price while
offering less availability. **Availability and family size turn on the same pre-result
number.**

**Why conditioning on it is legitimate:** `A0_planned_optimizer_steps` is derived from **the
same calibration artifact §2.6's own thresholds read**, it is a pre-result number **in exactly
the sense those thresholds are**, and it is **fixed before any arm produces an outcome**. A
family size chosen from that is **not** a family size chosen with results in hand.

When it activates, every exploratory contrast — A1, A2, A3 and both KTO arms — is
Holm-corrected over five. **A1–A3 are protected in the case where calibration comes back rich
and A5 never activates.** The confirmatory contrast and the structural secondary remain
families of one (§4.1).

#### A5.7 What this arm can and cannot show

**It moves objective, composition and dose together, so it can never attribute a positive
result to any one of them.** The write-up may not describe it as a dose experiment.

**Its value is asymmetric and real.** If A5 moves the endpoint, the endpoint is movable and
**A0's null is probably a dose problem**. If A5 is also null, the interpretation is
**bounded**: even five times the prompt coverage under a different objective failed to move
it. **Both outcomes are more informative than A0's null standing alone.**

**Planning inference, not a measurement.** At the pilot's yield A0 lands near **113 optimizer
steps**, so clearing 250 would require roughly **2.2× the pilot yield** — more than five
standard errors at n = 100. **A5 will therefore almost certainly activate.** Derivation:
post-screen population **11,071** (A3.3) × pilot `P_std = 39280300000/21820941 ≈ 1800.1194`
per 10,000 ≈ **1,992.9 projected pairs**; `ceil(1793/16) = 113`. **Every figure here is a
projection from a 100-prompt pilot, not a calibration measurement.**

#### A5.8 Everything else inherited

Model, revisions and LoRA from §3.4(a) · §3.4's pinned library table asserted for exact
equality · LR `5e-6`, cosine, `warmup_ratio 0.03`, `max_length 2048`, bf16, gradient
checkpointing on · `disable_dropout=True` · `precompute_ref_log_probs=False` ·
`sync_ref_model=False` · sequential sampling · seeds 42 · ref adapter parameter-hash equality
at step 0 · scoring per A4.8 with **MMLU excluded** for §4.5's reason · **one solo row per
admitted prompt, uncapped** except at most the A5.4 parity exclusion; composition is reported,
not constrained.

**Kill lines: §3.7 rules 1, 2, 4 and 5 as A4.7. Rule 3's replacement differs.**
A4.7's `eval_pair_reward_accuracy` reads only the pair rows, which are the **minority** here,
so it could kill this arm for failing at what it is not optimising.

> **Replaced by `eval_label_direction_accuracy` over *all* held-out rows:** a row counts toward
> the numerator iff `reward > 0` for a desirable row or `reward < 0` for an undesirable one,
> **ties count false**, reward as A4.7. **First-eval `>= 0.99` stops the arm.** Per-row values
> committed. `eval_pair_reward_accuracy` is still reported **as a diagnostic that kills
> nothing**.

#### A5.9 Timing, run order, and spend

**This amendment is adopted before the calibration run.** Launch eligibility then resolves
**mechanically from the calibration artifact**, never from another arm's result. **If both KTO
arms become eligible the run order is fixed here: A0 → A4-kto → A5-kto-wide.**

Adds to §3.11: *one wide KTO arm — 1 epoch over the expanded set at 32 unpaired rows per
optimizer step, plus `L` dev looks × 258 greedy generations.* **This amendment authorizes no
spend.**
