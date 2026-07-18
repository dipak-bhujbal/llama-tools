# AI/ML Glossary

Working reference for the terms you'll encounter across Weeks 1-12. Definitions are deliberately plain-English and short. "Why we care" notes appear on terms that are load-bearing for this project.

Grouped by area. Terms cross-reference where helpful.

---

## Model architecture

**Transformer** — the neural network architecture behind every modern LLM. Processes sequences of tokens, uses *attention* to relate tokens to each other, and predicts the next token. Introduced in the "Attention Is All You Need" paper (2017); every model we touch is a transformer.

**Token** — a subword chunk of text. Models read and write tokens, not characters or whole words. "hello world" might be 2 tokens or 5 tokens depending on the tokenizer. Token counts determine cost, context length, and sometimes output quality.

**Tokenizer** — the component that maps text ↔ tokens. Every model ships with its own tokenizer; using the wrong one produces gibberish. HF handles this automatically when you load a model with `AutoTokenizer.from_pretrained(...)`.

**Chat template** — the model-specific format that wraps a conversation for the model to consume. Llama uses `<|begin_of_text|><|start_header_id|>user<|end_header_id|>...`; other families use different markers. If you feed a chat model raw text without the template, quality drops sharply. Handled by `tokenizer.apply_chat_template(...)`.

**Attention** — the mechanism that lets each token look at every other token in the sequence when producing its next-token prediction. The reason transformers are good at long-range dependencies. Comes in variants: MHA (multi-head), GQA (grouped-query, used by Llama-3), MQA (multi-query).

**Context length** — the maximum number of tokens a model can see at once. Llama-3.1 is 128K tokens. Longer contexts cost more compute and memory.

**Parameter** — a number the model has learned during training. When we say "Llama-3.1-8B has 8 billion parameters," that's how many numbers the model stores. More parameters generally = better model + more compute needed to run it.

**Base model vs. Instruct model** — the *base* model is trained on raw text prediction and is not conversational. The *Instruct* variant has been additionally trained (SFT + often DPO/RLHF) to follow instructions and hold conversations. We use `Llama-3.1-8B-Instruct` as our base — the "Instruct" suffix matters.

**KV cache** — during generation, the model caches computed attention key/value tensors so it doesn't recompute them for prior tokens on every new token. Makes generation much faster at the cost of memory. Relevant to serving (Week 11).

---

## Training and post-training

**Pretraining** — training a model from randomly initialized weights on massive text corpora (trillions of tokens, millions of dollars). We do NOT do pretraining. Meta pretrained Llama-3.1.

**Fine-tuning** — updating an already-pretrained model's weights on a smaller, task-specific dataset (thousands to millions of examples, hundreds to thousands of dollars). We DO fine-tune. Fine-tuning is what turns a base model into an Instruct model, and what turns an Instruct model into a specialized model like our tool-calling variant.

**SFT (Supervised Fine-Tuning)** — the simplest form of fine-tuning: show the model chosen input → output pairs, minimize the loss between its predictions and the desired outputs. Our Week 4 is a full SFT run.

**DPO (Direct Preference Optimization)** — a technique for teaching a model to *prefer* certain outputs over others using preference pairs (chosen vs. rejected). Simpler and cheaper than PPO/RLHF, no reward model needed. Our Week 6.
- **Why we care:** DPO is the second stage of our two-stage post-training pipeline (see ADR-002). SFT teaches the model correct outputs; DPO teaches it to avoid plausible-but-wrong outputs.

**DPO beta** — a hyperparameter controlling how much DPO training is allowed to deviate from the reference model. Lower beta = more aggressive preference-following, higher beta = stays closer to the reference. Common values: 0.1 (default), 0.3, 0.5. Our Week 8 ablates over these.

**Reference model** — in DPO, a fixed model that anchors the trained model's behavior. Typically the SFT checkpoint. Prevents the DPO-trained model from drifting too far from what SFT learned.

**ORPO / KTO / SimPO / GRPO** — variants of preference-optimization techniques. Newer than DPO, sometimes simpler or more sample-efficient. We use DPO for maturity + defensibility (ADR-002); these are noted as future work.

**PPO / RLHF** — earlier preference-optimization methods that train a separate *reward model* and use reinforcement learning to update the trained model. More complex and expensive than DPO. Anthropic and OpenAI famously use RLHF variants; we don't need it for our project.

**Loss** — a number that measures how wrong the model is on a training example. Training minimizes loss over time. When we say "loss curve," we mean the plot of loss over training steps.

**Overfitting** — when a model learns the training data too specifically and generalizes badly to new inputs. Signal: train loss keeps dropping but eval loss stops dropping or rises. We use a held-out eval set to detect this.

**Epoch** — one full pass through the training dataset. We typically train for 1-3 epochs — more than that risks overfitting.

**Batch / batch size** — the number of examples processed together in one training step. Larger batches use more memory but train faster per epoch. We use batch 8 for SFT, 4 for DPO on A100 40GB.

**Learning rate** — how big a step the optimizer takes when updating weights. Too high → training explodes; too low → training doesn't converge. LoRA learning rates are typically 1000x higher than full-parameter ones because you're moving fewer weights.

**Gradient accumulation** — a trick to simulate a larger batch size when memory doesn't allow it: accumulate gradients across several small batches before applying an update.

**Mixed precision** — training with a mix of fp16/bf16 (fast, low memory) and fp32 (accurate, high memory) numbers. bf16 is the modern default for training on A100/H100.

---

## Efficient fine-tuning

**LoRA (Low-Rank Adaptation)** — instead of updating all model parameters, train a small *adapter* (usually ~1% of the total parameter count) that modifies the model's behavior. Much cheaper and faster than full fine-tuning; quality is often nearly identical for a wide range of tasks.
- **Why we care:** we use LoRA for both SFT (Week 4) and DPO (Week 6). Full-parameter fine-tuning of an 8B model would blow our $1000 compute budget.

**LoRA rank** — the "width" of the adapter matrices. Higher rank = more capacity, more compute. Common values: 8, 16, 32, 64. We start at rank 64.

**LoRA alpha** — a scaling factor applied to the adapter output. Convention is often alpha = 2 × rank, but this is heuristic.

**Adapter** — a small trainable module that modifies a pretrained model's behavior without changing the original weights. LoRA produces adapters. Adapters can be merged into base weights or kept separate.

**PEFT (Parameter-Efficient Fine-Tuning)** — the umbrella term for techniques like LoRA that fine-tune only a small fraction of parameters. The HF library `peft` implements them.

---

## Data

**Preference pair** — two model outputs for the same prompt: a *chosen* one (better) and a *rejected* one (worse). DPO consumes these. We synthesize ~15K for Week 3.

**Adversarial synthesis** — generating deliberately-wrong outputs (for use as *rejected* in preference pairs) by perturbing correct outputs. Cheaper than collecting real human failure examples.

**Deduplication** — removing near-duplicate examples from a dataset. Prevents the model from memorizing repeated examples and inflates apparent quality.

**MinHash + LSH** — a fast algorithm for detecting near-duplicates at scale. We use it to dedupe training data against the BFCL eval set (Week 3).

**Data leakage** — when training data contains examples from the eval set. Inflates eval scores meaninglessly. Aggressive dedup against eval sets prevents this.

**Chat template formatting** — applying the model's chat template to raw conversation data. Skipping this is a common bug — the model trains on data it will never see at inference time.

---

## Evaluation

**Benchmark** — a fixed dataset of tasks used to compare models on a specific capability. Public benchmarks let researchers compare results reproducibly.

**BFCL (Berkeley Function Calling Leaderboard)** — the canonical benchmark for function-calling / tool-use capability. Four categories: simple, parallel, multiple, multi-turn. Our target benchmark for Week 7.
- **Why we care:** this is what the resume line's "before → after" number will be measured on.

**MMLU (Massive Multitask Language Understanding)** — a broad general-knowledge benchmark. We use it as a *regression guardrail* to detect if SFT+DPO destroyed general capability.

**LLM-as-judge** — using one LLM to score another LLM's outputs. Cheap but has systematic biases (position bias, verbosity bias, self-preference bias). release-kit's eval-harness includes bias correction for this.

**Position bias** — LLM judges tend to prefer the first option shown when comparing pairs. Fix: swap A/B and average.

**Verbosity bias** — LLM judges tend to prefer longer responses. Fix: length-normalize or use a calibrated rubric.

**Regression suite** — a set of held-out tests you run before every release to detect quality regressions. release-kit's Pillar 1 includes this.

---

## Inference and serving

**Inference** — running a trained model to generate outputs. Different from training. Way cheaper.

**Serving** — hosting a model for real-time inference at scale. Concerns: throughput (tokens/sec), latency (time-to-first-token), cost, concurrency.

**vLLM** — a fast, open-source serving library for LLMs. Uses tricks like PagedAttention + continuous batching to maximize throughput. Our Week 11 benchmark.

**Quantization** — reducing the numerical precision of model weights (e.g., fp16 → int4). Trades a small amount of quality for major reductions in memory and compute. We do AWQ int4 in Week 9.

**AWQ (Activation-aware Weight Quantization)** — a specific quantization method that considers which weights are activated most often during inference. Higher quality than naive quantization at the same bit width.

**GPTQ** — an alternative quantization method. Slightly different quality/speed tradeoffs than AWQ. We defer GPTQ to a stretch goal.

**Calibration data** — a small sample of representative inputs used during post-training quantization to preserve quality on the tasks that matter. We'll use held-out tool-calling examples.

---

## Tools and ecosystem

**HuggingFace (HF)** — the GitHub of AI models. Hosts models, datasets, and libraries. Our primary distribution platform.

**HuggingFace Hub** — the actual site (huggingface.co) where models and datasets live.

**Model card** — the README for a model on HF, describing what it is, training details, evaluation results, intended use, and limitations. We write two (Week 10).

**transformers** (library) — HF's core library for loading and running models. What we use in `smoke.py`.

**TRL (Transformer Reinforcement Learning)** — HF's library for post-training loops (SFT, DPO, PPO). What we use for Weeks 2, 4, 6, 8.

**PEFT** (library) — HF's library for parameter-efficient fine-tuning (LoRA and friends).

**datasets** (library) — HF's library for loading, transforming, and pushing datasets to the Hub.

**wandb (Weights & Biases)** — experiment tracking service. Logs training curves, hyperparameters, sample outputs across runs. Free tier is fine for us.

**Runpod** — a cloud GPU rental service. Community spot A100 40GB @ ~$0.44/hr is our primary training environment.

---

## Project-specific glossary

**llama-tools** — this project. Fine-tuned Llama-3.1-8B for tool-calling.

**release-kit** — sibling project. Open framework for eval-gated LLM releases. llama-tools serves as its reference implementation.

**ADR (Architecture Decision Record)** — a short doc capturing why a decision was made, what was considered, and consequences. Our ADRs live in `docs/decisions/`. They're the primary record of *why* the project looks the way it does; interviewers will read them.

**Mode A (learning-as-we-build)** — our working mode: every technical decision must be understood well enough to defend cold in a 45-minute interview. Slower initial pace, real credential.

**BFCL v3** — the specific version of the Berkeley Function Calling Leaderboard we target.

---

## What is deliberately NOT in this glossary

Some concepts you may encounter but don't need to master for this project:
- **Mixture of Experts (MoE)** — sparse activation architectures. Interesting but not relevant to our dense 8B target.
- **Speculative decoding** — inference speedup technique. Deferred to a possible v2.
- **Mechanistic interpretability** — analyzing what individual model components do. Fascinating but out of scope.
- **RLHF training details (reward hacking, KL divergence)** — DPO sidesteps most of this. Learn on-demand if you go deeper into RLHF later.

If you encounter unfamiliar terms during study, note them and I'll add to this glossary as we go.
