# Amendment 2 — yield denominator and function-name matching

**STATUS: ADOPTED 2026-08-05. Superseded by `docs/prereg-study2.md`.**

Adopted by the owner in #general msg 2075 and landed in the preregistration
itself as **Amendment 2**, with a row in that document's amendments table. Under
Ground Rule 8 a threshold counts as preregistered only once it is committed to
the preregistration — so `docs/prereg-study2.md` is the governing text and this
file is now the drafting record, retained to show what was proposed, what review
changed, and when.

**Status of data at time of adoption: no study-2 model output, evaluation
result, or probe score had been observed. No model had been run. Both the
`multiple` baseline and the mining yield were unmeasured.**

Authorization: owner, #general msg 2024 ("yes in principle, with one check on
the draft"), then msg 2075 ("Adopt it, with one wording tightening first").

**Change made at adoption (msg 2075).** A2.1's denominator was "prompts the
miner actually attempted". Attempts are the one thing the ledger does not
preserve — a crashed-and-reconciled prompt is attempted twice, tombstoned once,
mined once — so that wording would have deflated yield on exactly the runs that
needed crash recovery, which is the silent lean this amendment exists to remove,
applied to its own gate number. Replaced with the owner's text: unique
post-screen prompts bearing an active, non-tombstoned ledger record, recomputable
from the same artifact as the numerator. A2.3 also gained a scope note recording
that argument-level matching is a preregistration amendment too, not a scoring
detail.

---

## A2.1 The yield denominator is post-screen prompts

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

## A2.2 The Phase 2 gate reads the run artifact, not this document

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

## A2.3 Function names are matched exactly, as presented

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

## Checks requested by the owner (msg 2024)

- [x] Pins yield as pairs per post-screen prompt — §A2.1.
- [x] States the measured survival rate must accompany any conversion — §A2.1.
- [x] Phase 2 gate reads the committed run artifact, not the prereg's quotation
      of provisional numbers — §A2.2.
- [x] Scorer normalization rule pinned before the prereg freezes, with the
      `multiple` set checked — §A2.3.

## Corrections applied in review (msg 2045)

- [x] §A2.1 no longer describes the decontamination fraction as *measured*. It
      is unmeasured until the run artifact of §A2.2 exists; the denominator is
      pinned now precisely because that number does not yet exist.
- [x] §A2.1 defines the numerator as pairs retained by deterministic
      materialization from **active** ledger candidates, replacing the
      ambiguous "pairs admitted".
- [x] §A2.3 makes the rule executable. Exact `parsed_name == gt_name` does not
      enforce key/tool-list consistency, so the rule now has a preflight part
      that fails closed, implemented as `preflight_key_names()` and called
      before generation.

## Open dependency

§A2.2 cannot be satisfied until the decontamination run artifact is committed.
That is in scope for the mining rewrite and is not a blocker for adopting this
amendment — the rule can be pinned before the artifact exists, which is the
point of preregistering it.
