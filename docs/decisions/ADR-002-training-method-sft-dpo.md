# ADR-002: Training method — SFT followed by DPO

**Status:** Accepted
**Date:** 2026-07-18

## Context

Given a chosen base model, we need to select a post-training methodology to improve tool-calling behavior. The choice affects: what dataset shape we need to curate, compute cost, defensibility in an interview setting (Mode A requirement), and how the improvement story reads in the model card.

## Options considered

| Method | What it is | Compute | Data needed |
|---|---|---|---|
| **SFT only** | Supervised fine-tune on correct tool-call examples | Low | Chosen examples only |
| **SFT + DPO** | SFT, then offline preference optimization | Medium | Chosen + rejected pairs |
| **SFT + ORPO** | Combined SFT + preference optimization in one loss | Medium | Chosen + rejected pairs |
| **SFT + KTO** | Kahneman-Tversky optimization (single-example labels, no pairs) | Medium | Chosen or rejected labels (unpaired) |
| **SFT + PPO/RLHF** | Full RLHF with a learned reward model | High | Preference pairs + reward model training |

## Decision

**SFT followed by DPO** as a two-stage pipeline.

## Rationale

1. **SFT alone is insufficient for our goal.** SFT teaches the model to produce correct tool calls when shown examples, but doesn't strongly discourage plausible-but-wrong alternatives. The failure modes we care about (wrong function selected, malformed args, hallucinated tools) are best addressed by preference optimization that explicitly downweights the near-miss failures.
2. **DPO over PPO/RLHF for cost and defensibility.** DPO is offline (no reward model training, no online sampling), which dramatically reduces compute cost and moving parts. Fits the $1000 budget cleanly. Also easier to defend cold in an interview — DPO's derivation is compact and well-documented.
3. **DPO over ORPO/KTO for maturity.** DPO is the canonical, extensively-benchmarked method. TRL support is battle-tested. ORPO and KTO are legitimate newer alternatives, but for a resume-credential project where the artifact must be defensible and reproducible, canonical > novel. We'll acknowledge ORPO/KTO in the technical report as future work.
4. **Two-stage separation gives us intermediate artifacts.** Publishing an "SFT-only" checkpoint before the DPO stage produces two model versions with distinct behavior — the ablation story is stronger, and the incremental training report shows disciplined methodology rather than one big black-box run.

## Consequences

- We need to curate ~15K preference pairs, not just chosen examples. "Rejected" examples require adversarial synthesis (perturbing chosen examples to introduce realistic failure modes) plus real-world failure samples where obtainable.
- DPO's `beta` hyperparameter meaningfully affects results; ablation across `beta ∈ {0.1, 0.3, 0.5}` is scheduled in Week 8.
- Reference model choice (SFT checkpoint) is deliberate — using the SFT checkpoint as reference (rather than base) constrains DPO from moving too far from the SFT-learned behavior, which is what we want.
- Interview defense: we should be able to derive DPO's loss from the KL-regularized objective, explain why the reference model matters, and articulate when DPO fails (length bias, low-signal preferences, degenerate collapses). Learning ramp in Week 5 covers this.

## Alternatives rejected specifically

- **Rejected pure RLHF/PPO** as compute-prohibitive and unnecessary given DPO's success on similar tasks.
- **Rejected ORPO** despite conceptual elegance — less mature tooling and less standard for interview conversations.
- **Rejected SFT-only** as it doesn't teach the model to *avoid* near-miss failures, which is our primary failure category.

## Revisit trigger

Revisit if: (a) DPO training runs consistently produce degenerate collapses on our data, (b) preference pair curation proves infeasible within budget/time, (c) a newer method (GRPO, SimPO, KTO variants) shows compelling and mature benchmarks on function-calling specifically.
