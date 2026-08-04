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

## 2. Mining `[PENDING]`

Pool composition, stratification toward multi-tool prompts, sampling parameters, verifier
version, and the yield gate arithmetic. **Must be committed before the pilot runs.**

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
