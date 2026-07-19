# ADR-003: Source dataset selection for SFT — Hermes + xLAM

**Status:** Accepted
**Date:** 2026-07-18
**Decision:** Combine `NousResearch/hermes-function-calling-v1` and `Salesforce/xlam-function-calling-60k` as SFT sources, target ~15K clean examples after dedup and filtering.

## Context

We need a source (or combination) of public tool-calling datasets for the SFT stage of training. The choice affects: quality of the resulting model, diversity of tool schemas the model sees during training, normalization work required, and licensing.

Not applicable this ADR: the eval benchmark. That's BFCL v3 per ARCHITECTURE.md; datasets used for training must be deduplicated against it.

## Options considered

| Option | Sources | Pros | Cons |
|---|---|---|---|
| **A: Hermes only** | `NousResearch/hermes-function-calling-v1` | Single format, well-maintained, ShareGPT-style messages already | Least diverse; limits generalization to novel tool schemas |
| **B: Hermes + xLAM (recommended)** | `hermes-function-calling-v1` + `Salesforce/xlam-function-calling-60k` | Two distinct generation styles, more tool-schema diversity; xLAM has 60K available, filter to what we need | xLAM format is different — needs normalization to messages format |
| **C: Hermes + xLAM + Glaive** | All of B + `glaiveai/glaive-function-calling-v2` | Maximum diversity | Glaive uses a raw-string `chat` field with embedded tool markers — heaviest parsing tax; low return per hour of parsing work |
| **D: Only synthetic** | Generate all training data via Anthropic/OpenAI API | Cleanest control over format + quality | Cost + slower + reintroduces the risk of a single-model bias |

## Decision

**Option B: Hermes + xLAM.**

Target composition of the final SFT set (~15K examples after dedup):
- Hermes (`func_calling` config): ~7,500 examples
- xLAM: ~7,500 examples

Both are Apache-2.0 (Hermes) or Creative Commons (xLAM CC-BY-4.0) — commercial-use compatible with our Apache-2.0 release.

## Rationale

1. **Two-source diversity without a heavy parsing tax.** Hermes gives us a strong baseline of hand-curated conversational tool-use examples; xLAM adds Salesforce's synthetic-generation approach, which covers a different distribution of tool schemas (many single-turn API-call patterns Hermes underemphasizes).
2. **Format cost is manageable.** Hermes is already in a ShareGPT-style messages format we can consume with light field-renaming. xLAM has structured fields (`query`, `tools`, `answers`) that map cleanly to a messages representation with ~30 lines of conversion code. Glaive would require parsing raw strings with embedded XML-like markers — much more brittle and lower return.
3. **Filter to quality, not to headcount.** Neither dataset is small — we can be selective. After dedup against BFCL v3 (per ADR-005 pending) and quality filters, we truncate to 15K. If quality is lower than expected, we expand the pool before we degrade the filter.
4. **Reserves upgrade room.** If Week 4's SFT results are weak, adding Glaive (Path C) is a clear upgrade path without redesigning the pipeline.

## Consequences

- `data/assemble_sft.py` needs a source-specific loader for Hermes and one for xLAM, plus a shared normalization layer producing a common messages schema.
- Provenance metadata (`source`, `source_id`) must be preserved on every example so we can trace training-set composition and debug source-specific quality issues later.
- The final dataset manifest should record: source counts before dedup, after dedup, after quality filter, and the final split-into-messages counts.

## Alternatives rejected specifically

- **Glaive** (Option C) — rejected for v1 because of parsing cost. Kept as a Week 4 fallback if diversity is insufficient.
- **Synthetic-only** (Option D) — rejected because single-model synthesis reintroduces model-specific stylistic bias (the exact thing tool-calling models are trying to reduce). We'll use synthesis for *rejected* preference examples (ADR-004 pending), not for the SFT set.

## Revisit trigger

Revisit if: (a) Week 4 SFT results show clear undertraining or format-inflexibility that more data diversity would fix, (b) Hermes or xLAM changes license terms in a restrictive direction, (c) a new, cleanly-formatted public function-calling dataset emerges with meaningfully better coverage.
