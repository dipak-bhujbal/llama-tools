# llama-tools: 12-Week Build Plan

Ship target: full v1 release (SFT+DPO model + AWQ quantized variant + preference dataset + technical report) within 12 weeks. Assumes ~8 hrs/day availability, Mode A (learning-as-we-build), $1000 hard cap on cloud compute.

Every week has: **Learn** (Mode A concepts covered), **Build** (concrete deliverables), **Decide** (decisions to make), **Spend** (compute cost estimate), **Definition of done**.

---

## Week 1: Fundamentals + environment

**Learn:** Transformer architecture at a high level; tokens and tokenization; instruct vs. base models; fine-tuning vs. pretraining; HuggingFace ecosystem overview (models, datasets, transformers, TRL, PEFT).

See [`docs/learning/week-1-fundamentals.md`](./docs/learning/week-1-fundamentals.md) for the study ramp and canonical resources.

**Build:**
- Set up HuggingFace account (already: `centuriandip`) + generate API token.
- Set up Runpod account, add payment method, familiarize with the console.
- Local Python 3.11+ environment via `uv`.
- Repo skeleton committed to `github.com/dipak-bhujbal/llama-tools`.
- Smoke test: run inference on `meta-llama/Llama-3.2-1B-Instruct` locally, one prompt, print output.

**Decide:** Nothing significant this week — foundations only.

**Spend:** $5-10 (single small Runpod A100 spot session for a 30-min "hello world" fine-tune on 1B model, mostly for the experience of launching a rented GPU).

**Definition of done:** You can explain fine-tuning in your own words. You have working local inference. Your HF and Runpod accounts are live. Repo is on GitHub.

---

## Week 2: SFT fundamentals + small-scale smoke test

**Learn:** Supervised fine-tuning theory; LoRA vs full fine-tune; how to read training/eval loss curves; overfitting signals; chat templates.

**Build:**
- LoRA-SFT of Llama-3.1-8B-Instruct on a *tiny* slice (500 examples) of Glaive function-calling data. Runs in ~30-60 min on a single A100.
- Push the LoRA adapter to HuggingFace as a private repo (`centuriandip/llama-3.1-8b-tools-week2-smoke`).
- Load the adapter, run 5 hand-picked prompts, compare with base model output.

**Decide:** LoRA rank (default: 64), alpha (default: 128) — commit or defer decision to Week 4.

**Spend:** $15-25 (2-4 hours of A100 time).

**Definition of done:** You have completed one full SFT run end-to-end. You can read the loss curves and identify whether the run behaved sensibly. Adapter is on HF.

---

## Week 3: Data curation

**Learn:** Dataset quality principles; deduplication (MinHash); chat template formatting for Llama-3.1; why preference pairs need adversarial variety; how to detect data leakage between train and eval sets.

**Build:**
- Curate ~15K SFT examples from: Glaive function-calling, APIGen, xLAM function-calling-60k.
- Dedupe against BFCL v3 eval set to prevent leakage.
- Format all examples per Llama-3.1 chat template.
- Synthesize ~15K preference pairs: chosen = correct tool call, rejected = adversarially perturbed (wrong function, missing arg, malformed JSON, hallucinated tool).
- Publish the preference dataset as `centuriandip/tool-calling-preferences` on HF (public, Apache 2.0).

**Decide:** Ratio of adversarial-synthesis vs. real-failure rejected examples. Filtering thresholds.

**Spend:** ~$5 (mostly local + Colab free tier; synthesis via small local model or Anthropic API).

**Definition of done:** Two datasets on disk (SFT + preference), both leakage-checked against BFCL, preference dataset published to HF.

---

## Week 4: Real SFT run

**Learn:** Distributed training basics (DDP, FSDP conceptually); mixed precision; gradient accumulation; the difference between train loss and eval loss and what each tells you.

**Build:**
- Full LoRA-SFT of Llama-3.1-8B-Instruct on the curated 15K SFT set. ~6-8 hours on 1x A100 40GB.
- Track: train loss curve, eval loss curve (held-out 500 examples), sample outputs at every checkpoint.
- Merge LoRA adapter into base model weights (for ease of downstream use).
- Push merged SFT-only checkpoint to HF as `centuriandip/llama-3.1-8b-tools-sft` (public).

**Decide:** Whether to continue with LoRA-SFT into DPO or attempt a full-parameter SFT for the DPO base (defaults to LoRA for cost).

**Spend:** $60-100 (one full 8-hour run + possibly one re-run).

**Definition of done:** SFT model published to HF with training curves in a linked wandb/mlflow report. You can explain why the loss curves look the way they do.

---

## Week 5: DPO fundamentals + preference-data validation

**Learn:** DPO theory (derive the loss from the KL-regularized RLHF objective); the reference-model concept and why it matters; beta hyperparameter; length bias; verbosity bias; degenerate collapse modes; how DPO differs from PPO/RLHF.

**Build:**
- Manually spot-check 200 preference pairs from the Week 3 dataset for quality (chosen actually correct, rejected actually plausible-but-wrong).
- Filter out pairs where chosen and rejected differ only trivially (would give DPO nothing to learn from).
- Prepare final DPO-ready dataset.
- Run a *tiny* DPO smoke test (500 pairs, 30 min) to verify the training loop wires up correctly.

**Decide:** Final preference dataset size (should be ≥10K after filtering to be robust).

**Spend:** $15-20.

**Definition of done:** You can derive the DPO loss on a whiteboard and explain what beta controls. Preference dataset is filtered and validated. Smoke test succeeded.

---

## Week 6: DPO training

**Learn:** Reference model selection tradeoffs; the interaction between SFT and DPO; how to read DPO training dynamics (chosen reward, rejected reward, reward accuracy).

**Build:**
- Full DPO run: SFT checkpoint from Week 4 as both the trained model and the reference model (with LoRA adapter setup for the trained side). ~8-10 hours on 1x A100.
- Track: chosen reward, rejected reward, reward margin, reward accuracy across steps.
- Merge final adapter, push to HF as `centuriandip/llama-3.1-8b-tools` (the primary release model, public).

**Decide:** Whether to freeze beta=0.1 or defer to Week 8 ablation (defer).

**Spend:** $80-120.

**Definition of done:** Primary model published to HF. DPO training curves in a linked report. You can explain what "reward accuracy climbing to 80%+ then plateauing" means.

---

## Week 7: Evaluation (BFCL v3 + MMLU regression)

**Learn:** BFCL v3 methodology (all four categories); AST-based vs execution-based scoring; MMLU as a general-capability guardrail; why we need a regression check.

**Build:**
- BFCL v3 evaluation harness (this is the first meaningful component that will also be reused in `release-kit`).
- Score three models across all four BFCL categories: base Llama-3.1-8B-Instruct, SFT-only checkpoint, SFT+DPO checkpoint.
- Run MMLU on all three to verify SFT+DPO didn't collapse general capability.
- Produce a comparison table: model × category × score, plus MMLU delta.

**Decide:** Whether v1 numbers meet the "meaningful improvement" bar (target: +5-10 points overall BFCL, MMLU within 2 points). If not, decide next steps: more data, ablations, method change.

**Spend:** $40-60 (BFCL v3 involves running the model on a few thousand prompts across categories).

**Definition of done:** BFCL scores captured for all three models. MMLU regression numbers captured. Table committed to `eval/results/week-7.md`.

---

## Week 8: Ablations

**Learn:** How to design an ablation study; how to report negative results honestly; hyperparameter sensitivity.

**Build:**
- DPO ablation: beta ∈ {0.1, 0.3, 0.5}. Three additional DPO runs.
- Reference model ablation (if time and budget allow): DPO with base model as reference vs. SFT checkpoint as reference.
- Score all variants on BFCL v3.
- Pick winner; write up ablation results.

**Decide:** Final beta value. Whether to re-publish the primary model with the winning ablation config.

**Spend:** $120-180 (three DPO runs @ 8-10 hours each on A100).

**Definition of done:** Ablation results table produced. Winning config selected. Primary HF model updated if the winning config differs from Week 6.

---

## Week 9: Quantization

**Learn:** Quantization theory (int4 vs int8 vs fp16); AWQ vs GPTQ (methodology differences); calibration data for post-training quantization; quality degradation under quantization.

**Build:**
- AWQ int4 quantization via [AutoAWQ](https://github.com/casper-hansen/AutoAWQ). Calibration set: held-out tool-calling examples.
- Re-run BFCL v3 on the quantized model; compare with full-precision.
- Push to HF as `centuriandip/llama-3.1-8b-tools-awq` (public).
- Update primary model card with quantization details and quality delta.

**Decide:** Whether to also produce a GPTQ int4 variant (defer if AWQ quality is acceptable).

**Spend:** $20-30 (quantization is relatively cheap; BFCL re-run is the main cost).

**Definition of done:** Quantized variant published. BFCL delta reported (target: <2% degradation). Memory reduction reported (target: ~3-4x).

---

## Week 10: Model card, technical report, HF release polish

**Learn:** Model card conventions (HF template, uses/limitations/ethics sections); how to write a defensible eval methodology section; how to acknowledge failure modes honestly.

**Build:**
- Full HF model cards for both `llama-3.1-8b-tools` and `llama-3.1-8b-tools-awq`.
- `docs/technical-report.md` (~2000 words): problem statement, method, data curation, training methodology, results, ablations, honest failure analysis, cost breakdown, limitations.
- README.md polish across the repo.
- Update all placeholder links in README to real HF/dataset/report URLs.

**Spend:** ~$5.

**Definition of done:** Model cards live on HF, meet HF template standards. Technical report committed and linked. All external-facing text is publication-ready.

---

## Week 11: Serving benchmark + release-kit reference implementation

**Learn:** vLLM serving basics; throughput vs. latency; batching; how quantized models change serving profile.

**Build:**
- Deploy `llama-3.1-8b-tools-awq` on vLLM (single-GPU rented Runpod instance).
- Benchmark: tokens/sec throughput at concurrency 1, 8, 32; first-token latency; time-to-first-token distribution.
- Wire llama-tools as reference implementation of `release-kit`: eval harness config, release checklist instance, monitoring config, all in the llama-tools repo.

**Decide:** Whether serving numbers should be added to the model card (probably yes) and whether to publish a `serving-benchmark.md` (yes).

**Spend:** $15-25.

**Definition of done:** Serving benchmark documented. llama-tools is a working reference implementation of release-kit end-to-end.

---

## Week 12: Publish, announce, interview prep

**Learn:** How to present technical work publicly; how to answer common interview probes on your own project.

**Build:**
- Final HF release polish; ensure both model repos, dataset repo, and technical report are all clean and cross-linked.
- Announcement post (LinkedIn + optional dev.to/HF community post): focus on the problem (open-model tool-calling reliability) and the concrete result (BFCL delta), not on the technique.
- Submit to Awesome-LLM and any relevant curated lists.
- **Interview prep (Mode A payoff):** cold-defend every technical decision in mock format. Practice 45-minute walk-throughs of: base model choice, data curation methodology, why SFT+DPO, DPO beta ablation, quantization quality tradeoff, honest failure modes. This is the deliverable that converts the credential from "exists" to "usable."

**Spend:** $5.

**Definition of done:** Model is public. Announcement is live. You can hold a 45-minute technical conversation about every decision without notes.

---

## Summary

| Week | Focus | Spend |
|---|---|---|
| 1 | Fundamentals + env | $5-10 |
| 2 | SFT smoke test | $15-25 |
| 3 | Data curation | $5 |
| 4 | Real SFT run | $60-100 |
| 5 | DPO fundamentals + validation | $15-20 |
| 6 | DPO training | $80-120 |
| 7 | Evaluation | $40-60 |
| 8 | Ablations | $120-180 |
| 9 | Quantization | $20-30 |
| 10 | Model card + tech report | $5 |
| 11 | Serving + release-kit integration | $15-25 |
| 12 | Publish + interview prep | $5 |
| **Total** | | **$385-585** |

Buffer against $1000 cap: **$415-615** for reruns, mistakes, and unplanned exploration.

---

## Deferred items (explicitly out of scope for v1)

- Merged upstream kernel PR to vLLM (potential v2 companion project)
- Multi-language function-calling
- Larger base models (13B, 70B) — compute-prohibitive at $1000
- RLHF with a learned reward model (DPO is sufficient for v1)
- Speculative decoding or draft-model integration
- Multimodal / vision tool-calling
