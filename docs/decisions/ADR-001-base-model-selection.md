# ADR-001: Base model selection — Llama-3.1-8B-Instruct vs. Qwen2.5-7B-Instruct

**Status:** Accepted
**Date:** 2026-07-18
**Decision:** Llama-3.1-8B-Instruct

## Context

The project is a fine-tune targeting structured tool-calling. We need to pick a base open-weight model. The choice materially affects: eligible download traffic post-release (ecosystem gravity), how much headroom exists for improvement on BFCL, compatibility with our tooling stack (TRL, vLLM, AutoAWQ), and licensing.

Two credible candidates dominate this decision: **Llama-3.1-8B-Instruct** and **Qwen2.5-7B-Instruct**. A brief appendix covers other options that were considered and rejected.

## Head-to-head: Llama-3.1-8B-Instruct vs. Qwen2.5-7B-Instruct

| Dimension | Llama-3.1-8B-Instruct | Qwen2.5-7B-Instruct | Advantage |
|---|---|---|---|
| **HuggingFace ecosystem gravity** | Largest open-model ecosystem on HF; fine-tunes get automatic discovery via Llama family search | Growing quickly but smaller download share outside China | **Llama** |
| **Raw benchmark strength (general)** | Strong across MMLU, HumanEval, MATH | Slightly stronger on many benchmarks (currently a top open 7B) | **Qwen** |
| **Tool-calling baseline (BFCL)** | Moderate — real headroom to demonstrate improvement | Already strong — less room to demonstrate a delta | **Llama (for our purpose)** |
| **Licensing** | Meta Llama license — permits open weights, attribution requirements | Apache-2.0 — cleaner, more permissive | **Qwen** |
| **Tooling maturity** | Tier-1 target for TRL, vLLM, AutoAWQ, PEFT | Well-supported but occasionally lags on cutting-edge integrations | **Llama** |
| **Download compounding post-release** | Meaningfully higher expected downloads for a fine-tune, all else equal | Growing but currently behind for solo unaffiliated releases | **Llama** |
| **Story clarity for resume/interview** | "Improved Llama tool-calling by X points" is instantly legible to any AI hiring manager | "Improved Qwen tool-calling by X points" requires slightly more context | **Llama** |
| **Model size** | 8B parameters (slightly larger) | 7B parameters | Neutral (both fit our compute budget) |

## Recommendation and rationale

**Llama-3.1-8B-Instruct.**

Three factors dominate for this specific project:

1. **Ecosystem gravity.** For a resume-credential project where post-release download compounding is a bonus outcome, starting from Llama is a materially better bet. Llama-family fine-tunes get surfaced in HF search patterns that Qwen fine-tunes currently don't match.

2. **Improvement headroom.** Llama-3.1-8B-Instruct's tool-calling baseline is moderate — good enough not to trivialize the project, weak enough that improvements are visible in the numbers. Qwen2.5's stronger baseline compresses the "before → after" delta we can report in the model card, which weakens the story even if the technical work is identical.

3. **Tooling maturity.** TRL, vLLM, AutoAWQ, PEFT all treat Llama as a first-class target. Fewer undocumented quirks means less time debugging tooling and more time on the actual work — meaningful for a $1000 budget where debugging burns dollars.

**The counter-case for Qwen:** Qwen is technically the stronger base model on paper, and the Apache 2.0 license is cleaner. If the goal were "produce the highest-scoring 7B tool-calling model in absolute terms," Qwen would be the pick. Our goal is different — it's "produce a defensible, discoverable, career-relevant artifact that reads clearly to AI hiring managers." Llama serves that goal better.

## Consequences

- Model card must clearly identify the base model and cite Meta's Llama license terms; downstream users need the license correctly propagated.
- If Qwen ecosystem overtakes Llama in AI-engineering download share over the next 6-12 months, this decision may need revisiting for a v2.
- We should acknowledge Qwen2.5-7B in the technical report as a reasonable alternative base and note why we chose Llama specifically for this project's goals.

## Appendix: Other options briefly considered and rejected

- **Mistral-7B-Instruct-v0.3:** Has native function-calling support in the tokenizer, Apache 2.0. Rejected because ecosystem momentum has shifted away — smaller impact per download than Llama or Qwen currently.
- **Qwen2.5-Coder-7B:** Best coding baseline; would suit "tool-calling for coding agents" niche. Rejected because narrowing to coding-specific tool-calling shrinks target audience and adds a domain framing we've been deliberately avoiding.
- **Hermes-3, ToolACE, xLAM:** Rejected as bases — these are *already* tool-calling fine-tunes. Fine-tuning them further would look like ignorance of prior art. Compete with them, don't start from them.

## How to accept this ADR

Reply "accept ADR-001" (or "flip to Qwen") in chat. Once accepted:
- Status changes from "Proposed" to "Accepted."
- I'll commit the accepted version and lock the base model reference across ARCHITECTURE.md, PLAN.md, and the Week 1 learning ramp (no changes needed if Llama is accepted — everything is already written against Llama-3.1-8B).
- If flipped to Qwen: I'll update those three files plus this ADR (~15 minutes of edits).

## Revisit trigger

Revisit if: (a) Meta materially changes Llama licensing in a restrictive direction, (b) another open-weight family clearly overtakes Llama in tool-calling benchmarks and download share, (c) targeted role postings begin explicitly preferring specific base-model expertise.
