# Amendment 2 (DRAFT) — yield denominator and function-name matching

**STATUS: DRAFT. NOT ADOPTED. NOT PREREGISTERED.**

This file is a proposal awaiting owner approval. It is deliberately *not* in
`docs/prereg-study2.md` and *not* in that document's amendments table, because
under Ground Rule 8 a threshold only counts as preregistered once it is
committed to the preregistration itself — before the run it governs. On
approval this text moves into `docs/prereg-study2.md` as Amendment 2, with a
row in the amendments table citing the approving message.

**Status of data at time of drafting: no study-2 model output, evaluation
result, or probe score has been observed. No model has been run. Both the
`multiple` baseline and the mining yield are unmeasured.**

Authorization pending: owner, #general msg 2024 ("yes in principle, with one
check on the draft").

---

## A2.1 The yield denominator is post-screen prompts

**Problem.** Fixing the mining bug where `n_prompts` was loaded *before*
decontamination (review finding 7) changes what a projected yield means.
Decontamination removes a measured fraction of the pool, so "pairs per prompt"
is ambiguous between two denominators that differ by that fraction — and the
ambiguity leans whichever way flatters the Phase 2 gate.

**Pinned definition.**

> **yield = (mined pairs admitted) / (post-screen prompts mined)**
>
> The denominator is prompts drawn from the pool *after* the decontamination
> screen has run, i.e. prompts the miner actually attempted. It is never the
> requested `--n-prompts`, and never the pre-screen pool size.

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

**Pinned rule.**

> A predicted function name is correct if and only if it is **byte-identical to
> the name as presented to the model in that item's tool list**. No
> normalization is applied: no case folding, no whitespace stripping, no
> module-prefix stripping, and no matching on the unqualified tail.

This is what `eval/bfcl_simple.py` already does (`name_ok = parsed_name ==
gt_name`). The amendment states it so it cannot drift, and records the measured
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

**Consequence for a disagreeing key.** If a pinned answer key expects a name
that is *not* among the tools presented for that item, that item is
internally inconsistent and cannot be scored under this rule. Such an item is
reported as a **key defect**, with the discrepancy documented and filed
upstream — it is not silently graded either way, and it is not dropped from the
denominator without being named.

---

## Checks requested by the owner (msg 2024)

- [x] Pins yield as pairs per post-screen prompt — §A2.1.
- [x] States the measured survival rate must accompany any conversion — §A2.1.
- [x] Phase 2 gate reads the committed run artifact, not the prereg's quotation
      of provisional numbers — §A2.2.
- [x] Scorer normalization rule pinned before the prereg freezes, with the
      `multiple` set checked — §A2.3.

## Open dependency

§A2.2 cannot be satisfied until the decontamination run artifact is committed.
That is in scope for the mining rewrite and is not a blocker for adopting this
amendment — the rule can be pinned before the artifact exists, which is the
point of preregistering it.
