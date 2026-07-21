# ADR-008: DPO v2 negative result — ship SFT as final, close on-policy DPO

**Status:** Accepted
**Date:** 2026-07-20
**Decision:** Do not merge any DPO v2 checkpoint. The Week 4 SFT model (`centuriandip/llama-3.1-8b-tools-sft`) remains the shipped model. DPO v2 is closed as a documented negative result: on-policy hard-pair DPO trained cleanly on preference metrics but monotonically degraded held-out tool-use *and* general capability (MMLU). The `centuriandip/llama-3.1-8b-tools` repo (planned SFT+DPO) stays unpublished. The project ships SFT.

## Context

ADR-007 designed DPO v2 as the pre-registered follow-up to ADR-006 (DPO v1 negative result). The core hypothesis: rule-perturbed rejecteds in v1 were mechanically easy, so the optimizer degraded the policy rather than adding preference signal. On-policy rejecteds sampled from the SFT model's own failures should be *hard by construction* — sitting in high-probability regions of the policy, requiring real ranking work.

Stages 1–3 executed cleanly today (2026-07-20):

- **Stage 1** (`data/sample_failures.py`): 7,880 unique training prompts × K=4 samples at T=0.8 = 31,520 candidate rejecteds. Failure rate 8.7% (2,739 total: 37 invalid_json + 99 wrong_tool + 2,603 wrong_args). Substantially above the 5% floor.
- **Stage 2** (`data/build_dpo_v2_pairs.py`): 2,523 final pairs after dedup + per-prompt cap 2. Well above the ≥1,500 abort floor. Composition: 94% wrong_args (2,378), 4% wrong_tool (100), 2% invalid_json (45).
- **Stage 3** (`train/dpo_v2_full.py`): 150 optimizer steps (1 epoch, effective batch 16). All pre-registered mechanical aborts survived:
  - First-eval `rewards/accuracies` = 0.627 (well below the 0.99 easy-data abort — pairs were genuinely hard, v2 data hypothesis validated)
  - `rewards/chosen` climbed monotonically: +0.008 → +0.021 → +0.027 (never touched the −0.25 kill line — opposite of v1, which drifted to −0.53 by step 400)
  - Final eval: chosen +0.019, rejected −0.110, accuracies 0.730, margins 0.129, eval_loss 0.639 (down from baseline 0.693)

By every training-time signal, DPO v2 fixed the ADR-006 failure mode. The v2 hypothesis on the *training-time* side is validated: on-policy hard pairs prevent the runaway policy damage v1 showed.

## Stage 4 results (held-out evaluation)

### BFCL v4 simple_python (399 held-out prompts, greedy generation, exact-match arg-in-accepted-list scoring)

| candidate | overall | name_ok | args_ok | json_valid | delta vs SFT |
|---|---|---|---|---|---|
| sft | 369/400 (92.25%) | 400/400 | 369/400 | 400/400 | — |
| dpo-50 | 364/400 (91.00%) | 400/400 | 364/400 | 400/400 | −5 |
| dpo-100 | 363/400 (90.75%) | 400/400 | 363/400 | 400/400 | −6 |
| dpo-150 | 359/400 (89.75%) | 400/400 | 359/400 | 400/400 | **−10** |

All candidates: perfect `name_ok` (function selection) and `json_valid` (structural output). Every failure is an `args_ok` failure. **DPO v2 monotonically degrades argument correctness with training steps.** Absolute single-checkpoint deltas are near the 95% CI noise band (±~2.7% on n=400), but the monotonic trend across three checkpoints is inconsistent with random noise and consistent with a small real effect.

### MMLU (14,042 items, 5-shot, next-token log-prob argmax over ' A'/' B'/' C'/' D')

| candidate | mmlu 5-shot | delta vs base | delta vs SFT |
|---|---|---|---|
| base (Llama-3.1-8B-Instruct) | 0.683 (full 14K) | — | — |
| sft | 0.659 (full 14K) | **−2.4** | — |
| dpo-50 | 0.658 (full 14K) | −2.5 | **−0.001** |
| dpo-100 | not measured on full 14K (see note) | ≈ dpo-50 | ≈ 0 |
| dpo-150 | not measured on full 14K (see note) | ≈ dpo-50 | ≈ 0 |

**Note on dpo-100 / dpo-150**: full-14K MMLU was run only for base + sft + dpo-50 (~90 min per candidate on the A40). dpo-100 and dpo-150 were not measured on the full test set because:

1. **dpo-50 came out indistinguishable from SFT** (−0.001), so the "DPO adds MMLU cost" hypothesis was already falsified by the first DPO checkpoint
2. **A prior 200-item smoke** across base/sft/dpo-50/dpo-100/dpo-150 showed all three DPO checkpoints scored *identically* (128/200 each), i.e. no progressive drift across DPO training steps. The smoke's absolute values were noisy (base 0.720 vs full-run 0.683) but the plateau finding is direct evidence that additional DPO steps do not shift MMLU further
3. Continuing would have added ~3 hrs of pod compute for a foregone-conclusion measurement. Cut on cost-per-bit grounds and disclosed here

**Findings — MMLU dimension**:
- SFT alone dropped MMLU by 2.4 pts vs base — squarely in the conventional "safe band" (~2-3 pts) for a domain-specialized fine-tune. Comparable to published specialized Llama fine-tunes (Code Llama, WizardMath: −3 to −8 pts). The SFT specialization cost is **modest and disclosable**, not catastrophic
- DPO v2 added essentially zero MMLU cost on top of SFT (0.658 vs 0.659). Consistent with the interpretation that DPO's parameter updates were confined to the tool-call generation distribution and did not touch the knowledge/reasoning circuits MMLU measures

## Options considered

### Option A: Merge dpo-50 (least-degraded on BFCL, MMLU-neutral)

Rejected. dpo-50 is measurably worse than SFT on the target task (−1.25 pts BFCL simple_python) and indistinguishable from SFT on general capability (−0.001 MMLU). Shipping "SFT+DPO" on the label with a measured tool-use regression inside — for zero gain anywhere — is a false claim we would then need to defend in a technical review. Same reasoning as ADR-006 Option A.

### Option B: Run more BFCL categories (`multiple`, `parallel`, `parallel_multiple`) hoping DPO wins somewhere

Rejected as a decision input. Additional categories would refine the *scope* of the negative result but cannot rescue the ship decision — DPO v2 already lost on `simple_python` (the closest category to our training distribution) with no offsetting gain elsewhere measured. Running more categories to search for a win is p-hacking. They are worth running as *documentation* of scope and are queued as an optional weekend follow-up, but the decision does not wait on them.

### Option C: Retune v2 — smaller learning rate, fewer steps, different beta

Rejected. The BFCL trajectory says the *first* DPO checkpoint (step 50) already degrades tool-use vs SFT; the MMLU trajectory says the *first* DPO checkpoint has already spent whatever capacity it was going to spend, with no further shift at 100/150 steps (smoke plateau). There is no training length or LR at which this data-recipe adds value — the direction is set at step 50. Same reasoning that closed v1's "retune don't rebuild" option in ADR-006.

### Option D (chosen): Ship SFT as final; close DPO v2 as a negative result

The SFT model wins outright on the target-task held-out (BFCL simple_python) and ties DPO on general capability (MMLU). It is already merged, signed, and released (Week 4). No further compute is required. The negative result is documented (this ADR), and the training-time success of ADR-007's design (mechanical aborts + on-policy hard pairs) is preserved as a methodology contribution independent of the model outcome.

## Decision

Option D. The shipped model is and remains `centuriandip/llama-3.1-8b-tools-sft`. `centuriandip/llama-3.1-8b-tools` (the planned SFT+DPO repo) stays unpublished.

## Root cause / interpretation

Three independent measurements form a consistent picture:

1. **Training-time preference metrics improved as designed.** ADR-007's design fixed the ADR-006 failure mode: chosen reward stayed positive, margins tripled, easy-data abort was not triggered. On-policy hard pairs *are* harder to rank than rule-perturbed pairs, and the optimizer *did* real ranking work. The training-side hypothesis is validated.

2. **The improved preference-ranking signal did not transfer to held-out generation quality.** BFCL simple_python argument-correctness monotonically dropped across the three checkpoints (−5, −6, −10 vs SFT). The most plausible interpretation is that on-policy DPO taught the model to discriminate *within-pair* on the specific failure modes we sampled, without generalizing to fresh held-out prompts — and the tiny distributional shift required to widen the training margins came at a small but measurable cost to correct outputs on new prompts.

3. **The MMLU regression check confirmed the damage was isolated to tool-use, not general capability.** SFT alone dropped MMLU by 2.4 pts vs base (0.683 → 0.659) — modest, inside the ~2-3 pt "safe band" for a domain-specialized fine-tune. DPO v2 added essentially zero on top (0.658 vs 0.659, delta 0.001). Neither the DPO reward-margin metrics nor a preference-set held-out could rule out capability regression — only an *independent* general-capability benchmark could. That check ran; it came back clean. This is a design win: **the guardrail was in place and it produced its evidence**, letting us reject DPO v2 on target-task grounds alone without confounding "did we also break something else" ambiguity.

The known limitation recorded in ADR-007 ("pair count is bounded by the model's failure rate — we get however many hard pairs exist, not a target count. That is the point.") was the right call for v2 — it made the training clean and the failure analyzable — but "hard on the training distribution" turned out not to imply "helpful on the held-out distribution." That gap between training-signal and held-out-utility is the load-bearing finding of Week 7 and generalizes beyond this project: on tasks with objective ground truth where SFT is already near-ceiling, DPO's preference-ranking signal has little to add and non-zero risk of shifting the model away from correct outputs. This ADR should be cited if a future project considers DPO for a similar objective-answer task.

## Consequences

- `centuriandip/llama-3.1-8b-tools` stays unpublished. The project's final shipped model is the SFT-only model.
- The three DPO v2 checkpoints (50/100/150) and the final best-model adapter are archived at [`centuriandip/llama-3.1-8b-tools-dpo-v2-evidence`](https://huggingface.co/datasets/centuriandip/llama-3.1-8b-tools-dpo-v2-evidence) as evidence, not candidates. Same handling as ADR-006 v1 checkpoints.
- `eval/out/bfcl_simple/` and `eval/out/mmlu_regression/` artifacts (raw generations + per-item MMLU predictions + report tables) are archived at the same HF dataset repo (`eval/bfcl_simple/`, `eval/mmlu_regression/`) so a reviewer can independently re-score any candidate without re-running inference.
- `train/merge_dpo_winner.py` is not run. It remains valid for a future v3 if one is designed.
- README and model card must present the DPO v1 + v2 sequence as a two-stage negative result, not omit either. The methodology (pre-registered aborts, on-policy hard-pair sampling, independent held-out eval catching what training gates missed) is the primary artifact of Week 7.
- The `eval/bfcl_simple.py` + `eval/mmlu_regression.py` harnesses are kept — they are release-kit's first eval-harness contribution (planned reuse per `eval/README.md`).
- The Week 7 evaluation story ("pre-registered gates → training passed → held-out eval → honest verdict → close v2") joins ADR-006 as evidence that the project's evaluation methodology works.

## Revisit trigger

Revisit only if:

1. Independent held-out re-scoring of the BFCL simple_python sweep is shown to have mis-graded — e.g., the arg-in-accepted-list scorer diverges from BFCL's official AST scorer in a way that flips more than 5 items and reverses the sign of the SFT-vs-DPO delta. (The current scorer is intentionally stricter than BFCL's, so it under-credits all candidates equally; a re-score would likely *widen* SFT's lead, not narrow it.)
2. A v3 preference set is designed to explicitly target held-out generalization (e.g., temperature-sampled rejecteds on prompts the SFT model has never seen), *and* passes an independent pre-registered gate that includes MMLU regression before training launches.

Neither trigger is planned. The project ships SFT.
