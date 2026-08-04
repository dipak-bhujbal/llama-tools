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
that made it — before the affected run starts.

| Date | Section | Change | Reason | Commit |
|---|---|---|---|---|
| — | — | — | — | — |
