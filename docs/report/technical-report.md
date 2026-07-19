# Technical Report: Fine-Tuning Llama-3.1-8B-Instruct for Tool-Calling via SFT + DPO

**Status:** Skeleton — fill as runs complete. Target: Week 9-10.
**Source-of-truth order:** week-4-run-log.md (and future run logs) > trainer_state.json > wandb > memory.

---

## Abstract

_2-4 sentences: problem statement, method in one breath, headline result, one honest limitation. Write last._

Llama-3.1-8B-Instruct demonstrates moderate tool-calling reliability on BFCL v3; this project applies SFT on {{sft_example_count}} curated tool-calling examples followed by DPO on {{dpo_pair_count}} adversarially-generated preference pairs to improve structured output fidelity. Training was conducted on a single RTX A6000 48GB GPU (Runpod) within a $1000 compute budget.

**Placeholder numbers:**
- Overall BFCL v3 delta (base → SFT+DPO): {{bfcl_overall_delta}} points
- Total compute spend: ${{total_spend}}

---

## 1. Motivation

_Explain why open-model tool-calling reliability matters, why the improvement is measurable (BFCL v3 exists), and why this project is scoped the way it is (Llama family, 8B scale, $1000 cap). 2-3 paragraphs when filled._

Key points to cover:
- Tool-calling as a capability gap in open-weight models vs. proprietary APIs
- Why Llama-3.1-8B-Instruct was chosen over Qwen2.5-7B-Instruct (ecosystem gravity, improvement headroom, tooling maturity) — full rationale in **ADR-001** (`docs/decisions/ADR-001-base-model-selection.md`)
- Why SFT+DPO over SFT-only or RLHF — full rationale in **ADR-002** (`docs/decisions/ADR-002-training-method-sft-dpo.md`)
- Project scope: resume-credential artifact targeting Sr AI TPM/PM interviews; defensibility of every decision is a first-class requirement

---

## 2. Data

### 2.1 SFT Dataset

_Describe the two source datasets, the assembly pipeline, the dedup run, and the final split. Numbers are exact — lift from `docs/progress/weeks-1-3.md` (Week 3 section) and the run log config table._

**Sources** (ADR-003: `docs/decisions/ADR-003-source-datasets.md`):

| Source | Loaded | Kept after dedup | Share of final |
|---|---|---|---|
| NousResearch/hermes-function-calling-v1 | 3,786 | 1,161 | ~10% |
| Salesforce/xlam-function-calling-60k | 11,214 | 10,999 | ~90% |
| **Total** | **15,000** | **12,160** | — |

**Dedup:** MinHash + LSH, Jaccard threshold 0.7, 128 permutations, word-level 5-grams. Hermes drop rate 69% reflects genuinely templated `func_calling_singleturn` queries, not a dedup bug. Evidence: `docs/progress/weeks-1-3.md` Week 3 dedup section.

**Train/eval split:** 11,660 train / 500 eval, seed 42. Split performed in `train/sft_full.py` @ commit e350f16.

**BFCL leakage dedup:** deferred to Week 7 concurrent with eval harness build. Rationale documented in `data/dedupe.py`. [UPDATE THIS SECTION IF LEAKAGE WAS FOUND]

**Figures:**
- [FIGURE: source composition pie chart — Hermes vs. xLAM pre- and post-dedup]

### 2.2 Preference Dataset

_Describe preference pair construction: which source subset, the 5 perturbation types, why rule-based over LLM-generated, the 1:1 ratio decision. Numbers from `weeks-1-3.md` Week 3 preference section and ADR-004._

**Source subset:** xLAM subset only (Hermes skipped — multi-turn prose structure makes reliable perturbation brittle). Rationale: ADR-004 (`docs/decisions/ADR-004-preference-synthesis.md`).

**Perturbation taxonomy:**

| Type | Count | Share |
|---|---|---|
| malformed_json | 2,401 | 21.8% |
| wrong_arg_value | 2,361 | 21.5% |
| missing_required_arg | 2,329 | 21.2% |
| hallucinated_tool | 2,318 | 21.1% |
| wrong_tool_from_list | 1,590 | 14.5% |
| **Total (pre-filter)** | **10,999** | — |

**Triviality filtering** (ADR-005: `docs/decisions/ADR-005-preference-pair-triviality-filtering.md`):

| Signal | Pairs removed |
|---|---|
| near_identical_edit_distance | 683 |
| exact_match | 59 |
| canonical_json_equal | 59 |
| exempt_unparseable_rejected (spared) | 2,394 |
| **Final DPO set** | **10,243** |

Key design decision: `malformed_json` pairs (one-character JSON corruptions) are exempt from the edit-distance filter because textual proximity does not imply semantic proximity — a missing closing brace is a total parse failure, not a minor variation. Implemented in `data/validate_preferences.py :: triviality_signals()`.

Artifact: `centuriandip/tool-calling-preferences` on HuggingFace (private until release).

### 2.3 Quality Audit and the xLAM Python-Expression Bug

_This section covers the mid-run data-quality incident found during the 200-pair spot-check on 2026-07-19. This is a feature of the report, not a footnote — it demonstrates the quality audit process and the principled decision not to restart._

**Incident summary:** During the 200-pair spot-check (model-assisted triage + human adjudication), 2 of 200 pairs had corrupt `chosen` examples. Root cause: upstream xLAM annotation bug — argument values written as Python expressions (`"[0.02] * 5"`) instead of literal JSON arrays. This is valid Python but not valid JSON and would cause a runtime type error in any calling environment.

**Scope check:** Full dataset audit revealed 15 DPO pairs and **16 SFT training targets** (of 11,660, 0.14%) carry the bug. The SFT run (Week 4) was already in progress when the bug was found.

**Decision: do not restart.** 0.14% contamination over 3 epochs cannot meaningfully shift an 8B model; restart cost was ~2 hours of GPU time and repeat spend. Decision rationale documented in `docs/progress/week-4-run-log.md` (10:35 entry).

**Mitigations applied:**
1. `corrupt_chosen_python_expr` filter added to `validate_preferences.py`
2. DPO set re-generated: 10,999 → 10,243 (16 corrupt pairs purged)
3. SFT set cleanup queued for any future re-run
4. Qualitative sample check: flag any outputs containing Python-expression argument values

**Triage audit results:** 194/200 ok, 2 bad-chosen (the bug), 3 trivial (already caught by automated filter — validates filter is working), 1 content-skipped.

**Evidence:** `docs/progress/week-4-run-log.md` entry at 10:35.

---

## 3. SFT Method and Training Run

_Describe the SFT setup: LoRA rationale, config table, hardware, cost. Numbers are exact from the run log config table and early metric entries. Most of this section can be lifted directly._

### 3.1 Method

**LoRA vs. full fine-tune:** LoRA chosen for cost (fits in A6000 48GB with gradient checkpointing; full fine-tune of 8B in bf16 requires ~64GB for weights alone). LoRA rank 64 / alpha 128 is the standard high-capacity configuration for instruction-following tasks. Decision in ADR-002.

**Chat template:** Llama-3.1 native template (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`). PAD token set to EOS (token 128009 = `<|eot_id|>`) — standard for Llama-3.1 which ships without a dedicated pad token. Tokenizer alignment warning is benign; documented in run log (09:26 entry).

### 3.2 Training Configuration

Lifted verbatim from `docs/progress/week-4-run-log.md` config table (commit e350f16):

| Field | Value |
|---|---|
| Script | `train/sft_full.py` @ commit e350f16 |
| Base model | meta-llama/Llama-3.1-8B-Instruct (bf16) @ `0e9e39f249a16976918f6564b8830bc894c89659` |
| Data | `sft_dedup.jsonl` — 11,660 train / 500 eval (seed 42) |
| LoRA | r=64, alpha=128, dropout 0.05, q/k/v/o projections |
| Schedule | 3 epochs, 1,095 steps, lr 2e-4 cosine, 3% warmup (~33 steps) |
| Batch | per-device 8 × grad accum 4 = effective 32, max seq 2048 |
| Memory | gradient checkpointing on, bf16 |
| Hardware | 1× RTX A6000 48GB, Runpod @ $0.49/hr |
| Checkpoints | every 200 steps, keep 3; eval every 100 steps |

### 3.3 Training Dynamics

Early metrics (from `docs/progress/week-4-run-log.md` 09:23 entry):

| step | loss | grad_norm | lr | token_acc |
|---|---|---|---|---|
| 10 | 1.192 | 0.741 | 5.46e-05 | 0.759 |
| 20 | 0.751 | 0.337 | 1.15e-04 | 0.831 |
| 30 | 0.635 | 0.780 | 1.76e-04 | 0.851 |
| 40 | 0.561 | 0.261 | 2.00e-04 | 0.868 |

Step 100 eval: eval_loss 0.4625, eval token accuracy 0.8859 (from run log 09:51 entry).
Step 200 eval: eval_loss 0.4016, eval token accuracy 0.8978 (from run log 10:52 entry).

**Final metrics (fill from `outputs/sft-full/trainer_state.json`):**

| Metric | Value |
|---|---|
| Final train loss | {{final_train_loss}} |
| Final eval loss | {{final_eval_loss}} |
| Final eval token accuracy | {{final_eval_token_accuracy}} |
| Best checkpoint step | {{best_checkpoint_step}} |
| Wall clock | {{sft_wall_clock_hrs}} hrs |
| Total cost | ${{sft_cost}} |

**Figures:**
- [FIGURE: train/eval loss curves across all 1,095 steps — from wandb or `trainer_state.json` log_history]
- [FIGURE: eval token accuracy across checkpoints]

**Base model revision:** `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f249a16976918f6564b8830bc894c89659` (HF main as of 2026-07-19). This revision must be pinned in the model card and all reproduction instructions.

### 3.4 SFT Artifact

HuggingFace repo: `centuriandip/llama-3.1-8b-tools-sft` (public).
Revision: {{sft_hf_commit_sha}}.
Merge status: LoRA adapter merged into base weights [YES/NO — fill after merge].

---

## 4. DPO Method and Training Run

_Placeholder section — fill during Week 6. Structure mirrors Section 3._

### 4.1 Method

DPO loss derivation sketch (fill in Week 5 learning ramp): optimizes KL-regularized RLHF objective without a reward model by expressing the optimal policy in closed form and reparameterizing the reward in terms of policy log-ratios. Key hyperparameter: beta controls KL penalty strength.

Reference model: SFT checkpoint (not base model) — constrains DPO from unlearning SFT-trained formatting. Rationale: ADR-002.

### 4.2 Training Configuration

[PLACEHOLDER — fill from Week 6 run log]

| Field | Value |
|---|---|
| Script | `train/dpo.py` @ commit {{dpo_commit}} |
| Base/reference model | `centuriandip/llama-3.1-8b-tools-sft` @ {{sft_hf_commit_sha}} |
| Data | `preferences_dpo.jsonl` — 10,243 pairs |
| LoRA | {{dpo_lora_config}} |
| Beta | {{dpo_beta}} (winning value from Week 8 ablation) |
| Hardware | {{dpo_hardware}} |
| Cost | ${{dpo_cost}} |

### 4.3 Training Dynamics

[PLACEHOLDER — fill from Week 6 run log]

Key metrics to report:
- Chosen reward trajectory across steps
- Rejected reward trajectory across steps
- Reward margin (chosen − rejected)
- Reward accuracy (fraction of steps where chosen reward > rejected reward)

**Figures:**
- [FIGURE: chosen/rejected reward curves — from wandb DPO run]
- [FIGURE: reward accuracy across steps]

### 4.4 Beta Ablation

[PLACEHOLDER — fill from Week 8 ablation results]

Beta values tested: {0.1, 0.3, 0.5}. Winning value: {{dpo_beta_winner}}.

| Beta | BFCL overall | Reward accuracy | Notes |
|---|---|---|---|
| 0.1 | {{bfcl_beta_01}} | {{reward_acc_beta_01}} | |
| 0.3 | {{bfcl_beta_03}} | {{reward_acc_beta_03}} | |
| 0.5 | {{bfcl_beta_05}} | {{reward_acc_beta_05}} | |

Evidence: `eval/results/week-8.md` (to be created).

### 4.5 DPO Artifact

HuggingFace repo: `centuriandip/llama-3.1-8b-tools` (public, primary release).
Revision: {{dpo_hf_commit_sha}}.

---

## 5. Evaluation

### 5.1 BFCL v3 Setup

_Describe the eval harness, the four BFCL categories, how scoring works, and the leakage check. Fill during Week 7._

BFCL v3 categories:
- Simple function calling (AST-based)
- Multiple functions (AST-based)
- Parallel functions (AST-based)
- Relevance detection (binary — should the model call a function at all?)

Leakage check: MinHash dedup of SFT training set against BFCL v3 test prompts. [STATUS: deferred to Week 7 — results will appear here. If leakage found, describe remediation.]

Eval harness: `eval/bfcl_harness.py` @ commit {{eval_harness_commit}}.
BFCL v3 version: {{bfcl_version_tag}}.

### 5.2 Results

[PLACEHOLDER — fill from `eval/results/week-7.md`]

| Model | Simple | Multi-func | Parallel | Relevance | Overall |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct (base) | {{base_simple}} | {{base_multi}} | {{base_parallel}} | {{base_relevance}} | {{base_overall}} |
| + SFT | {{sft_simple}} | {{sft_multi}} | {{sft_parallel}} | {{sft_relevance}} | {{sft_overall}} |
| + SFT + DPO | {{dpo_simple}} | {{dpo_multi}} | {{dpo_parallel}} | {{dpo_relevance}} | {{dpo_overall}} |

Delta (base → SFT+DPO): {{bfcl_overall_delta}} points overall.

**Figures:**
- [FIGURE: BFCL v3 category-by-category bar chart, 3 models side by side]

### 5.3 General Capability Regression (MMLU)

[PLACEHOLDER — fill from Week 7]

MMLU scores: base {{mmlu_base}}, SFT {{mmlu_sft}}, SFT+DPO {{mmlu_dpo}}. Delta vs. base: {{mmlu_delta}} points. Target: within 2 points of base. [PASS/FAIL against target]

---

## 6. Qualitative Analysis

_Present concrete model outputs — both good and bad — for representative tool-calling scenarios. This section requires human judgment, not just metrics. Pull examples from `outputs/sft-full/sample_generations.json` and the Week 4 run log end-of-run checklist._

### 6.1 Success Cases

[PLACEHOLDER — 3-5 examples where the model produces a correct, well-formed tool call]

Example format for each:
- User query
- Available tools (abbreviated)
- Model output
- Assessment: why this is correct

### 6.2 Failure Cases

[PLACEHOLDER — 3-5 examples where the model fails, with failure taxonomy]

Priority failure modes to document:
- Python-expression argument values (related to the xLAM bug — does the model emit `"[0.02] * 5"` style values? See run log 10:35 decision)
- Tool hallucination (calls a tool not in the provided list)
- Argument structure errors (right tool, wrong arguments)
- Missing tool call (should call a tool, produces prose instead)

### 6.3 DPO Effect on Failure Modes

[PLACEHOLDER — compare SFT-only vs SFT+DPO outputs on the same failure-prone prompts]

Per-perturbation-type reward accuracy from DPO training will indicate which failure modes the model learned to reject:

| Perturbation type | DPO reward accuracy |
|---|---|
| malformed_json | {{reward_acc_malformed}} |
| wrong_arg_value | {{reward_acc_wrong_arg}} |
| missing_required_arg | {{reward_acc_missing_arg}} |
| hallucinated_tool | {{reward_acc_hallucinated}} |
| wrong_tool_from_list | {{reward_acc_wrong_tool}} |

---

## 7. Limitations

_Be explicit about what this model cannot do and where the methodology has gaps. Honest limitations increase interview defensibility, not decrease it._

**Known limitations to document (expand when filling):**

1. **Single GPU, single run:** No hyperparameter search at full scale. LoRA rank 64 / alpha 128 is a reasonable default, not a tuned optimum.

2. **xLAM-dominated training set:** Post-dedup composition is ~90% xLAM / 10% Hermes. The model has seen disproportionately little of conversational multi-turn tool-use patterns. Generalization to Hermes-style dialogue is not guaranteed. Evidence: `docs/progress/weeks-1-3.md` dedup analysis.

3. **Rule-based rejected samples only:** DPO rejected examples were generated by 5 deterministic perturbation rules. Real model failure modes are more diverse. An attacker who reads ADR-004 can construct inputs that evade the trained-to-reject patterns.

4. **0.14% SFT corruption:** 16 of 11,660 SFT training targets contained Python-expression argument values from xLAM upstream. Effect is expected to be negligible at this scale but cannot be zeroed.

5. **No BFCL v3 leakage analysis yet:** Deferred to Week 7. If contamination is found, evaluation numbers require a caveat. [UPDATE THIS WHEN LEAKAGE CHECK COMPLETES]

6. **MMLU as regression proxy:** MMLU is a broad academic benchmark, not a reliable measure of tool-calling-adjacent reasoning degradation. A task-specific regression benchmark would be more informative.

7. **No live execution eval:** BFCL v3 AST-based scoring checks structural correctness, not execution correctness. A model can score well by producing correctly-shaped JSON that would still fail at runtime (wrong semantics, type mismatches).

---

## 8. Lessons Learned

_Seed entries are real incidents from the lab notebook. Expand with any additional lessons before publishing._

### 8.1 Infrastructure

**Pod disk wipe on stop:** Runpod container disk does not persist when a pod is stopped (only volume storage persists). Week 4 pod required a full fresh setup on resume: clone, pip install, HF login, dataset re-pull. Mitigation: all datasets pushed to HF before stopping; training script is idempotent from clone. Cost: ~1 hour of setup time at start of Week 4. Evidence: `docs/progress/week-4-run-log.md` ~09:00 entry.

**Lesson:** Never stop a Runpod pod without first verifying that all in-progress artifacts are pushed to HF or an attached volume. Add a pre-stop checklist to the repo.

### 8.2 Ecosystem Versioning

**`push_to_hub` version drift:** TRL 1.8's `trainer.push_to_hub()` failed with `create_model_card() got an unexpected keyword argument 'repo_id'` due to bundled transformers version mismatch. Fixed by switching to `HfApi.create_repo` + `upload_folder`, which is stable across ecosystem versions. Evidence: `docs/progress/weeks-1-3.md` Week 2 lessons section.

**Lesson:** Do not use TRL convenience upload methods in production scripts. Prefer direct HF API calls with explicit version pinning. Pin `transformers`, `trl`, `peft`, `accelerate` versions in `requirements.txt` immediately after a working run.

### 8.3 Data Quality

**Filter measures text, not semantics:** The triviality filter's edit-distance signal correctly removed 705 `wrong_arg_value` pairs where a numeric value differed by a trivial amount (e.g., `"5"` vs `"5.0"`). But it would have incorrectly removed 2,394 `malformed_json` pairs — one-character JSON corruptions — because the text is nearly identical even though the semantic difference is maximal (valid executable call vs. total parse failure). Required a selective exemption (ADR-005). Evidence: `docs/decisions/ADR-005-preference-pair-triviality-filtering.md`.

**Lesson:** When filtering synthetic data, always verify that the filter's measurement proxy (textual distance, embedding similarity, etc.) actually captures the semantic property you care about. Run a stratified spot-check per perturbation type before finalizing the filter.

**xLAM Python-expression annotation bug:** Upstream xLAM dataset contains argument values written as Python expressions (`"[0.02] * 5"`) rather than evaluated literal values. These are valid Python but not valid JSON. Discovered mid-run via the 200-pair spot-check. Affected 0.14% of SFT set and 0.14% of preference set. Evidence: `docs/progress/week-4-run-log.md` 10:35 entry.

**Lesson:** Always run a sample-based quality audit before training, not during. The 200-pair audit was planned for Week 5 but was moved up; running it before the Week 4 run would have caught the bug in time to fix the SFT set.

[PLACEHOLDER — add lessons from DPO and eval runs as they accumulate]

---

## 9. Reproducibility

_Everything needed to reproduce this work from scratch, with exact versions and costs._

### 9.1 Code

| Component | Repo | Commit |
|---|---|---|
| Training pipeline | `github.com/dipak-bhujbal/llama-tools` | {{repo_head_commit}} |
| SFT script | `train/sft_full.py` | e350f16 |
| DPO script | `train/dpo.py` | {{dpo_script_commit}} |
| Data assembly | `data/assemble_sft.py`, `data/assemble_preferences.py` | {{data_scripts_commit}} |
| Eval harness | `eval/bfcl_harness.py` | {{eval_harness_commit}} |

### 9.2 Model Revisions

| Artifact | HuggingFace repo | Revision SHA |
|---|---|---|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` |
| SFT checkpoint | `centuriandip/llama-3.1-8b-tools-sft` | {{sft_hf_commit_sha}} |
| SFT+DPO (primary) | `centuriandip/llama-3.1-8b-tools` | {{dpo_hf_commit_sha}} |
| AWQ quantized | `centuriandip/llama-3.1-8b-tools-awq` | {{awq_hf_commit_sha}} |
| Preference dataset | `centuriandip/tool-calling-preferences` | {{dataset_hf_commit_sha}} |

### 9.3 Environment

| Package | Version |
|---|---|
| Python | 3.13 |
| torch | 2.13.0 |
| transformers | 5.14.1 |
| trl | {{trl_version}} |
| peft | {{peft_version}} |
| accelerate | {{accelerate_version}} |
| huggingface_hub | 1.24.0 |

Hardware: 1× RTX A6000 48GB on Runpod (`cautious_maroon_loon` pod; pod ID is ephemeral — hardware spec is what matters for reproduction).

### 9.4 Compute and Cost

| Stage | Hardware | Wall clock | Cost |
|---|---|---|---|
| Weeks 1-3 (local + smoke) | Local + Runpod A6000 | ~13m (smoke) | $0.17 |
| Week 4 SFT | Runpod A6000 @ $0.49/hr | {{sft_wall_clock_hrs}} hrs | ${{sft_cost}} |
| Week 6 DPO | {{dpo_hardware}} @ ${{dpo_rate}}/hr | {{dpo_wall_clock_hrs}} hrs | ${{dpo_cost}} |
| Week 7 eval | {{eval_hardware}} | {{eval_wall_clock_hrs}} hrs | ${{eval_cost}} |
| Week 8 ablations (3 DPO runs) | {{ablation_hardware}} | {{ablation_wall_clock_hrs}} hrs | ${{ablation_cost}} |
| Week 9 quantization | {{quant_hardware}} | {{quant_wall_clock_hrs}} hrs | ${{quant_cost}} |
| **Total** | | | **${{total_spend}} of $1000 cap** |

### 9.5 Determinism Notes

- All data splits use seed 42 (set in `data/assemble_sft.py` and training script).
- MinHash dedup uses 128 permutations, Jaccard threshold 0.7; output is deterministic given the same input order.
- Perturbation types for preference pairs use seed 42 in `data/assemble_preferences.py`.
- Training is not bit-for-bit deterministic across GPU runs (CUDA non-determinism in attention) but hyperparameters and data are fully reproducible.

---

## Appendix A: Architecture Decisions Index

| ADR | Decision | Status |
|---|---|---|
| ADR-001 | Base model: Llama-3.1-8B-Instruct over Qwen2.5-7B | Accepted |
| ADR-002 | Training method: SFT + DPO | Accepted |
| ADR-003 | SFT sources: Hermes + xLAM | Accepted |
| ADR-004 | Preference synthesis: rule-based adversarial perturbation | Accepted |
| ADR-005 | Triviality filter: exempt malformed-JSON pairs from edit-distance signal | Accepted |

Full ADR texts: `docs/decisions/`.

---

## Appendix B: Figures Checklist

Fill before finalizing:

- [ ] [FIGURE: train/eval loss curves, all 1,095 SFT steps] — source: wandb or `trainer_state.json :: log_history`
- [ ] [FIGURE: eval token accuracy across SFT checkpoints]
- [ ] [FIGURE: source composition pre/post dedup (pie or stacked bar)]
- [ ] [FIGURE: DPO chosen/rejected reward curves]
- [ ] [FIGURE: DPO reward accuracy across steps]
- [ ] [FIGURE: BFCL v3 results bar chart, 3 models × 4 categories]
- [ ] [FIGURE: beta ablation BFCL scores]
- [ ] [FIGURE: MMLU regression bar chart]
