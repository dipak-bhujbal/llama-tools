# Week 4 run log — full SFT

Live lab notebook for the Week 4 training run. Timestamps are local (2026-07-19).

Notebook discipline: every entry timestamped; past entries are never edited — corrections are appended as new entries.

## Config (as launched)

| Field | Value |
|---|---|
| Script | `train/sft_full.py` @ commit e350f16 |
| Base model | meta-llama/Llama-3.1-8B-Instruct (bf16) |
| Data | `sft_dedup.jsonl` — 11,660 train / 500 eval (seed 42) |
| LoRA | r=64, alpha=128, dropout 0.05, q/k/v/o projections |
| Schedule | 3 epochs, 1,095 steps, lr 2e-4 cosine, 3% warmup (~33 steps) |
| Batch | per-device 8 × grad accum 4 = effective 32, max seq 2048 |
| Memory | gradient checkpointing on, bf16 |
| Hardware | 1× RTX A6000 48GB, Runpod pod `cautious_maroon_loon`, $0.49/hr |
| Checkpoints | every 200 steps, keep 3; eval every 100 steps |

## Timeline

- **~09:00** — pod resumed (container disk had been wiped on stop; fresh setup: clone, pip install, HF login, dataset pulled from private HF repo `centuriandip/llama-tools-sft-data`).
- **~09:01** — training launched inside tmux session `train`. Model download 16.1GB, tokenize 11,660 examples ~12s. Two benign warnings: `warmup_ratio` deprecation; PAD/EOS aligned to token 128009 (standard Llama-3.1).
- **09:03** — step 4/1095, 29.1 s/it. Initial ETA ~8.8 hrs.
- **09:23** — step 49/1095, 26.1 s/it (ETA revised ~7.6 hrs). Logged metrics (steps 10/20/30/40):

| step | loss | grad_norm | lr | entropy | token_acc |
|---|---|---|---|---|---|
| 10 | 1.192 | 0.741 | 5.46e-05 | 0.869 | 0.759 |
| 20 | 0.751 | 0.337 | 1.15e-04 | 0.761 | 0.831 |
| 30 | 0.635 | 0.780 | 1.76e-04 | 0.637 | 0.851 |
| 40 | 0.561 | 0.261 | 2.00e-04 | 0.562 | 0.868 |

  Reading: steep early drop (format learning), token accuracy already 87%, grad norms small and stable, warmup completed on schedule at ~step 33 — peak lr reached, cosine decay begins. Week 2 smoke comparison: 1.48 → 0.80 on 500 examples; lower start here expected (same distribution, more data).

- **09:26** — provenance + decision-point entry (run 1 baseline):
  - Base model revision: `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f249a16976918f6564b8830bc894c89659` (HF main as of today; the pod downloaded this revision at launch).
  - Dataset composition (pre-dedup → final): Hermes function-calling 3,786 + Salesforce xLAM 11,214 = 15,000 → MinHash+LSH dedup (threshold 0.7) → 12,160 (Hermes dropped ~69% — templated queries; see ADR-003). Split 11,660 train / 500 eval, seed 42.
  - Tokenizer warning, verbatim: `[transformers] The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 128009, 'pad_token_id': 128009}.`
  - Decision on the warning: accepted as benign. `sft_full.py` sets `pad_token = eos_token` because Llama-3.1 ships without a pad token; 128009 is `<|eot_id|>`, the standard end-of-turn token for this family. Transformers aligning model/generation config to the tokenizer is the *desired* direction. Verify at generation time that outputs terminate on `<|eot_id|>` correctly before trusting sample quality.
  - The step 10-40 metrics table above (09:23 entry) is the **run 1 baseline** — any re-run gets compared against it.

## End-of-run checklist (to fill before touching DPO)

- [ ] Final train loss / eval loss / token accuracy at step 1,095 (from `trainer_state.json`).
- [ ] Full eval_loss trajectory (steps 100–1,000) — flag any upward divergence and which checkpoint was best.
- [ ] Qualitative generation samples — `outputs/sft-full/sample_generations.json` (auto-generated) PLUS manual reads: include both good and bad examples in this log. Token accuracy can be high while tool calls are subtly wrong (right format, wrong function/args) — catch that here, not after DPO.
- [ ] Verify generations terminate cleanly on `<|eot_id|>` (closes out the PAD/EOS decision above).
- [ ] Honest verdict, written here: ready for DPO, or needs another SFT pass? (Criteria: eval loss stable, samples produce correct tool selection and argument structure on the majority of spot-reads.)

- **09:51** — step ~100 check-in (epoch 0.27, 9% done). Steps 50-100 train loss plateauing into the grind phase: 0.503 / 0.484 / 0.471 / 0.507 / 0.461 / 0.484 (10-step logs), token acc 0.876 → 0.888, grad norm stable 0.20-0.25, cosine decay begun (lr 1.981e-4). **First eval point, step 100: eval_loss 0.4625, eval token accuracy 0.8859** — eval loss at-or-below train loss ⇒ no overfitting signal, model is generalizing. Context: Week 2 smoke finished at eval_loss 0.897 on 500 examples; we're at 0.4625 with 91% of training still to go. Eval pass itself took 128s (500 examples).

- **10:35** — data-quality incident found MID-RUN via the 200-pair spot-check (model-assisted triage + human adjudication). Triage of the preference sample: 194/200 ok, 2 bad-chosen, 3 trivial, 1 content-skipped. The 2 bad-chosen share a root cause: upstream xLAM annotation bug — argument values written as Python expressions (`"[0.02] * 5"`) instead of literal JSON arrays. Checked the FULL datasets: 15 DPO pairs and **16 SFT training targets** (of 11,660) carry the bug — i.e. the currently-running SFT job is training on 16 corrupt targets (0.14%).
  - **Decision: do NOT restart the run.** 0.14% of examples cannot meaningfully shift an 8B model over 3 epochs; a restart costs ~2 hrs of progress + repeat spend for negligible quality gain. The 16 examples teach a malformed arg-value style at worst.
  - **Mitigations:** (a) `corrupt_chosen_python_expr` filter added to `validate_preferences.py`; DPO set re-generated: 10,999 → 10,243 final (16 corrupt purged). (b) SFT set cleanup queued for any future SFT re-run. (c) Watch for Python-expression arg values in tonight's qualitative samples — if the model emits them, this line item explains why.
  - Also confirmed working as designed: the 3 trivial pairs the triage caught were in the *pre-filter* sample by intent (sample audits the filter itself); all 3 are ones the automated filter already removes.

- **10:52** — step ~220 check-in (epoch 0.60, 21% done). **Eval, step 200: eval_loss 0.4016** (from 0.4625 at step 100 — down 0.061, healthy slope), eval token accuracy 0.8978. Train loss now in 0.39-0.42 band, token acc crossed 0.90, entropy 0.40, grad norms 0.22-0.28 (one benign 0.40 blip ~step 170). Eval still tracking train loss with no divergence. lr decayed to 1.87e-4.

- **11:36** — step ~310 check-in (epoch 0.85, 28% done). **Eval, step 300: eval_loss 0.3436** (0.4625 → 0.4016 → 0.3436; steps of −0.061, −0.058 — still near-linear improvement, deceleration not yet visible), eval token accuracy 0.9107. Train loss 0.34-0.39 band, token acc ~0.91, grad norms 0.22-0.32, lr 1.71e-4. Still zero train/eval divergence.

- **12:24** — false alarm resolved: a 12:23 re-paste was stale scrollback; live screenshot confirms 409/1095 (37%), 27.6 s/it, ETA ~5h15m. **Eval, step 400: eval_loss 0.2967**, eval token accuracy 0.9224. Trajectory 0.4625 → 0.4016 → 0.3436 → 0.2967 (−0.061/−0.058/−0.047 — first sign of deceleration). Crossed into epoch 2 (epoch ~1.10): train loss stepped down into the 0.25-0.32 band as expected on repeat data — the train/eval gap (train ~0.28 vs eval 0.297) is now slightly positive, normal for epoch 2; watching for it to widen materially, which would be the overfit signal.

- **12:58** — step 477 (44%, epoch 1.29), 27.8 s/it, ETA ~4h45m. Epoch-2 train loss settled into 0.24-0.30 band, token acc 0.92-0.94, grad norms 0.25-0.35 (marginally up vs epoch 1 — normal at lower loss). No new eval since step 400; next at 500.

- **13:17** — step 515 (47%, epoch 1.40). **Eval, step 500: eval_loss 0.2640**, eval token accuracy 0.9304. Trajectory 0.4625 → 0.4016 → 0.3436 → 0.2967 → 0.2640 (deltas −0.061/−0.058/−0.047/−0.033 — clean deceleration curve, textbook cosine). Train 0.21-0.30, acc up to 0.944; train/eval gap still small and stable — no overfit signal at mid-epoch-2.

- **14:01** — step 600 (55%, epoch 1.65). **Eval: eval_loss 0.2400**, eval token accuracy 0.9368. Deltas now −0.061/−0.058/−0.047/−0.033/−0.024. Train 0.21-0.26, acc ~0.94; grad norms easing back (0.20-0.31); lr 8.97e-5 (past halfway of cosine). Gap train≈0.23 vs eval 0.24 — still healthy.

- **15:02** — step 723 (66%, epoch 1.97). **Eval, step 700: eval_loss 0.2248**, eval token accuracy 0.9407. Deltas: −0.061/−0.058/−0.047/−0.033/−0.024/−0.015. Train 0.19-0.25, acc ~0.95. Note: pace slowed 28.4 → 35.4 s/it (likely host contention on the migrated pod); ETA pushed to ~6:40 PM. Epoch-2 boundary imminent → epoch-2-end checkpoint (~step 730) is the designated fallback if epoch 3 overfits.

- **15:42** — step 803 (73%, epoch 2.19, now in epoch 3 territory data-wise). **Eval, step 800: eval_loss 0.2191** (delta just −0.006 — nearly flat but still improving; no upturn). Eval acc 0.9425. Train loss dropped to 0.14-0.19 band with acc up to 0.96 — the train/eval gap is now clearly widening (train ~0.17 vs eval 0.219), classic epoch-3 memorization onset while eval still inches down. Steps 900/1000 decide final-vs-checkpoint. Pace degraded further: 46.6 s/it (28 → 35 → 47; host contention worsening); ETA now ~7:30 PM.

- **15:43** — config gap acknowledged (advisor catch): `load_best_model_at_end` was not set; cannot be added mid-run. Mitigation: manual best-checkpoint selection at merge time — read eval history from trainer_state.json, merge the surviving checkpoint (600/800/1000/final) nearest the best-eval step. Lesson for future runs: set `load_best_model_at_end=True, metric_for_best_model="eval_loss"` and align save/eval cadence from the start.

- **16:30** — step ~905 (82%, epoch 2.47). **Eval, step 900: eval_loss 0.2141** (−0.005; still improving, marginally). Eval acc 0.9437. Train loss stabilized 0.15-0.18 (not plunging further); train/eval gap holding ~0.045, not widening. The predicted ~epoch-2.5 plateau is materializing as a soft landing, not an upturn. lr down to 1.6e-5. Step-1000 eval decides the merge target.

- **17:21** — step 1000 (91%, epoch 2.74). **Eval, step 1000: eval_loss 0.2120** (−0.002; tenth consecutive improvement, never turned up). Eval acc 0.9444. Pace recovered to 27.9 s/it (contention cleared); ~44 min remain, ETA ~18:05. Full eval trajectory: 0.4625 / 0.4016 / 0.3436 / 0.2967 / 0.2640 / 0.2400 / 0.2248 / 0.2191 / 0.2141 / 0.2120.
  - **Merge-target decision (pre-registered):** step-1000 is the last scheduled eval (next would be 1100 > 1095 total). The final adapter sits ≤95 steps past the best measured point, taken at lr < 4e-6 (negligible movement), with the trend still downward at last measurement. Decision: **merge the FINAL adapter**. Rationale: best-measured checkpoint (1000) and final differ by ~95 near-zero-lr steps on a still-improving trend; no evidence of upturn anywhere in the run. If tonight's qualitative samples look off, checkpoint-1000 remains on disk as fallback.

- **18:14 — RUN COMPLETE. Final metrics (from trainer_state.json):**
  - Total runtime 32,962s = **9h 09m** (1,095 steps, 3.0 epochs); 1.061 samples/s; total FLOPs 1.86e18; cost ≈ $4.55.
  - **Final eval (step 1095, epoch 3.0): eval_loss 0.21165, eval token accuracy 0.9445** — an unscheduled end-of-training eval fired, and it was the 11th consecutive improvement (0.2120 → 0.2117). The pre-registered merge-the-final-adapter decision is vindicated by measurement: the final adapter IS the best measured point.
  - Whole-run average train_loss 0.2852; last logged train window (step 1090): loss 0.166, token acc 0.955, grad_norm 0.19, lr 1.6e-8.
  - Eval loss trajectory (steps 100→1095): 0.4625, 0.4016, 0.3436, 0.2967, 0.2640, 0.2400, 0.2248, 0.2191, 0.2141, 0.2120, 0.2117.
- **18:13 — qualitative sample review (5/5 read, gate PASSED):** 4/5 exact match to gold (country-info multi-call with param types; cricket+F1 multi-call; dividend+histogram multi-call; whois). 1/5 subtle argument miss: Brazilian health news requested, model emitted `"lr": "en-US"` vs gold `"pt-BR"` — right tool, right schema, wrong locale grounding. Zero malformed JSON, zero Python-expression artifacts (watch-item clean), multi-call handling flawless. The en-US/pt-BR miss is the canonical motivating example for the DPO stage (wrong_arg_value class, 1,654 pairs). Decision: ship final adapter; checkpoint-1000 fallback not needed.

<!-- Append merge/upload/sign-off outcomes below. Do not edit entries above. -->
