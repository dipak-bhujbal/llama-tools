# ADR-009: BOS handling — a confirmed train/eval asymmetry, disclosed and kept

**Status:** Accepted
**Date:** 2026-08-09
**Decision:** Keep the existing evaluation prompt construction, which feeds the model **two** `<|begin_of_text|>` tokens, and record it in every run manifest. Do not switch evaluation to single-BOS mid-study. The pinned SFT adapter was **trained** under a single-BOS construction and has been **evaluated** under a double-BOS one. That asymmetry **is correctable** — switching evaluation to single-BOS would align the baseline and the planned arms with their training construction — and is nonetheless **disclosed rather than corrected**, because doing so would place Study 2 in a different construction from Study 1 with no measurement connecting them. **Its effect on any reported figure is unmeasured.**

## Context

`eval/bfcl_simple.py::build_prompt` renders with `apply_chat_template(tokenize=False, add_generation_prompt=True)`. The returned **string already contains** `<|begin_of_text|>`. `generate()` then calls `tokenizer(prompt, return_tensors="pt")` without `add_special_tokens`, which resolves to `True` on the pinned tokenizer and prepends a **second** BOS.

This surfaced on 2026-08-09, when the §0 isolation ladder refused to run: it asserted the first production prompt was 609 tokens and the production path produced 610. The 609 corresponded to counting the same string with `add_special_tokens=False` — a path production never takes. The launcher exited **11 seconds** after start (`PROBE_EXIT_RECORD pid=1843 exit=69 elapsed=11s`); the **pod stayed allocated ~32 minutes** across setup, the refusal and teardown. Billing follows allocation, not run time, and any figure derived from elapsed × rate is **derived, not a settled charge**.

The preregistration (`docs/prereg-study2.md`) **specifies no tokenization path**. There is therefore nothing to amend; this ADR records a construction the protocol left open, in the same spirit as `BASE_CANDIDATE_REALIZATION` (see `eval/bfcl_simple.py`).

## Evidence

### The evaluation path has always been double-BOS

Verified across every commit touching `eval/bfcl_simple.py`, from its first (`00e43d3`, 2026-07-20) through the 2026-08-08 crash commit (`2d8abdb`) and `08be5d3` to the file's current version, last changed by `8963bca`: the render call passes `tokenize=False` and the generation call never sets `add_special_tokens`. `git diff 2d8abdb 08be5d3 -- eval/bfcl_simple.py` touches neither call; later commits changed the file without changing either call site.

Observed directly against the pinned tokenizer, offline:

```
first production prompt, category=multiple, id multiple_0
len(tokenizer(prompt))                            = 610
len(tokenizer(prompt, add_special_tokens=False))  = 609
leading token ids                                 = [128000, 128000, 128006]
                                                    <|begin_of_text|> ×2, <|start_header_id|>
```

### The SFT adapter was trained single-BOS — recovered receipt

The pinned adapter revision `b6f4da479f8c6fc044ee8b802a92f47780f970c5` publishes a TRL-generated model card at `adapter/README.md` (SHA-256 `ae486da3efafaad86b9074b438fb63365dca25c84126397815c7dfa880c0b426`):

```
### Framework versions
- PEFT 0.19.1
- TRL: 1.8.0
- Transformers: 5.14.1
- Pytorch: 2.8.0+cu128
- Datasets: 5.0.0
- Tokenizers: 0.22.2
```

`train/sft_full.py` at commit `e350f16` passes conversational `messages` to `SFTTrainer`, so TRL applies the chat template itself. Released TRL 1.8.0 then tokenizes with `add_special_tokens=False` — `trl/trainer/sft_trainer.py:684,722,729`, commented *"to avoid adding the BOS twice"* (`dpo_trainer.py:332,339,346` does the same). The archived `adapter/chat_template.jinja` (SHA-256 `e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65`) matches the pinned base tokenizer's template exactly and emits BOS.

**Calibration.** This holds *under the released TRL 1.8.0 implementation identified by the adapter receipt*. The receipt records a package **version**, not a hash of the installed `site-packages`; a locally patched installation is not directly excluded, though nothing suggests one.

### W&B recovery — a named negative result

The authenticated default entity contains only project `huggingface`, holding one run (`fwmqedp2` / `sft-smoke-week2`) and no full-SFT run. Project `llama-tools` does not exist. This is consistent with `train/sft_full.py`, which declares `WANDB_PROJECT` but never exports it, and enables `report_to` only when `WANDB_API_KEY` is present in the training process.

**Recorded as a specific failed search, not as proof that no run exists anywhere.**

## What is actually being traded

An earlier draft of this ADR claimed no option could align both sides. **The recovered SFT receipt disproves that**, and the claim is withdrawn rather than carried forward.

The SFT baseline is now **known** to have been trained single-BOS, and the registered Study-2 arm trainers (`SFTTrainer`, `DPOTrainer` in TRL 1.8.0) — which have not yet been run for Study 2 — also tokenize with `add_special_tokens=False`. So switching evaluation to single-BOS **would** align the current SFT baseline and the planned Study-2 arms with their training construction. That is a real alternative, not a relabelling.

The trade-off is therefore **cross-study construction continuity versus known train/eval alignment**:

- **Keeping double-BOS** preserves continuity with the construction that produced Study-1's evaluation figures, and accepts that the **SFT baseline and the planned SFT-derived arms** are evaluated outside the construction they were fine-tuned under. This says nothing about the `base` candidate: the input construction used in the base model's own pretraining and instruction-tuning is not established here.
- **Switching to single-BOS** would put the SFT baseline and planned arms in-regime, and would make new absolute figures non-comparable with `369/400`.

Study-1 DPO-v2's unrecovered training environment is a limitation on **retrospective interpretation of DPO-v2**, not a reason the SFT baseline and planned Study-2 arms could not be evaluated in the construction they were fine-tuned under.

**The effect of the retained asymmetry on the confirmatory contrast is unmeasured.** Direction and magnitude are both unknown; no items have been scored both ways. This ADR makes no claim that the effect is small, and none that it is large.

## Scope of the recorded label

`bos_handling` in the run manifest describes the **evaluation generation path only** — `eval/bfcl_simple.py::build_prompt → generate` — and its `applies_to` field says so in the artifact.

**`mining/backend.py:191` shares the same render-then-default-retokenize construction** and produced the pilot preference pairs, which are Study-2 **training data**. That path is deliberately outside the manifest label rather than silently covered by it. A wider audit of the ten call sites of `apply_chat_template` outside tests — four of which are training scripts — has not been performed.

## Options considered

1. **Keep the current construction, record and disclose it.** *(chosen)*
2. **Switch evaluation to single-BOS.** Would put the SFT baseline and the planned Study-2 arms in the construction they are trained under, at the cost of continuity with Study-1's evaluation construction and comparability of new absolute figures with `369/400`.
3. **Measure the effect, then decide.** Deferred — see the revisit trigger.
4. **Defer entirely.** Rejected: leaves the construction undocumented.

## Decision

Option 1, on the owner's continuity rationale: Study-1's figures were produced under this construction, and changing evaluation mid-study would mean every subsequent number carried a footnote about which regime produced it. This chooses cross-study continuity **over** known train/eval alignment, which option 2 could have delivered — it is not a choice forced by the absence of an aligned option.

## Consequences

- `eval/bfcl_simple.py` records `bos_handling` (expectations) in **every initial manifest**, and `bos_handling_observed` in **every run that reaches the preflight** — once the tokenizer has loaded and at least one prompt has been built, before any model loads. A run that dies earlier records the expectations without an observation.
- A run whose observed construction does not match the approved one **raises `BosHandlingMismatch` before loading a model**, leaving an `incomplete` manifest carrying the observation. Figures cannot be produced under an unapproved construction.
- Tests tie every recorded field to observable behaviour, so the label cannot drift from the code.
- Absolute figures — including Study 1's 369/400 (92.25%) — were produced under a construction that differs from the SFT adapter's training construction in this respect. **This is disclosed, not corrected, and its effect is unquantified.**

## Environment difference, recorded without attribution

Training ran **PyTorch 2.8.0+cu128** (adapter card); the 2026-08-08 and 2026-08-09 probe pods ran **2.9.1+cu128** (`pip_freeze.txt` under `llama-tools-artifacts/`). Disclosed as an environment difference between training and evaluation. **No causal claim is made**, in particular none connecting it to the 2026-08-08 CUDA fault.

## Revisit trigger

**Before publication, or publication-facing use, of any new Study-2 absolute figure.** (ADR-008 and this ADR already report `369/400`; the trigger concerns figures not yet produced.) The project's headline KPIs are absolute, and an absolute number produced under an input construction that differs from its model's training construction is exactly the condition a sensitivity study exists to address. Option 3 returns at that point and will require candidates, an item set, a primary endpoint, a paired analysis, a runtime, and its own cost estimate and approval.

Also revisit if: Study-1 DPO-v2's training environment is recovered; the base model revision changes; or any of the four training scripts is shown to use a different construction.
