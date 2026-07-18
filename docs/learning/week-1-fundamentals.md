# Week 1 — Fundamentals

This is your study ramp for Week 1. You've said you have zero prior AI background, so this document assumes nothing. By the end of the week you should understand the *concepts* well enough that Week 2's training code doesn't feel like magic.

Time budget: ~15-20 hours across the week (roughly half of a 40-hour week — the rest goes to environment setup and the smoke test).

---

## What you need to understand by end of Week 1

The following concepts should be things you can explain in your own words to a colleague:

- **What a transformer is (high level).** The architecture behind every modern LLM. You do not need to derive attention math this week — just understand: transformers process sequences of tokens, use attention to relate tokens to each other, and predict the next token.
- **Tokens and tokenization.** Text is chunked into "tokens" (subword pieces) before the model sees it. Different models use different tokenizers, and this affects everything downstream — context length, cost, sometimes output quality.
- **Base vs. Instruct models.** A "base" model is trained on raw text prediction and is not conversational. An "Instruct" model has been additionally trained to follow instructions and hold conversations. We're using `Llama-3.1-8B-Instruct` — the "Instruct" suffix matters and is why we don't need to teach the model conversational format from scratch.
- **Pretraining vs. fine-tuning.** *Pretraining* is training a model from random weights on massive text corpora (billions of tokens, millions of dollars). *Fine-tuning* is taking an already-pretrained model and updating its weights on a smaller, task-specific dataset (thousands of examples, hundreds of dollars). We are fine-tuning, not pretraining.
- **What LoRA is.** Low-Rank Adaptation. Instead of updating all ~8 billion parameters of the model (expensive), LoRA trains a small "adapter" (a few million parameters) that modifies the model's behavior. Cheaper, faster, and the adapter can be merged back into the model or kept separate.
- **HuggingFace ecosystem.** HuggingFace is the GitHub of AI models. Three things you'll interact with:
  - **`transformers`** (library) — loads models and runs inference.
  - **`datasets`** (library) — loads training data.
  - **`TRL`** (library) — training loops for SFT, DPO, PPO. This is what we'll use to fine-tune.
  - HuggingFace Hub is where models and datasets live (like GitHub for models).

---

## Recommended reading order (all free)

Do these roughly in this order. Skip past sections that get too math-heavy for now — you can come back later.

1. **[Andrej Karpathy: "Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY)** — 2-hour video. The single best introduction to how transformers work. You don't need to code along; just watch. Skip the tokenization deep-dive if it drags. *Why it matters:* by the end you'll understand what "attention" and "next-token prediction" actually mean, which is 60% of everything else.
2. **[HuggingFace NLP Course, Chapters 1-3](https://huggingface.co/learn/nlp-course)** — reading, ~2-3 hours. Introduces the `transformers` library, tokenization, and the model loading pattern you'll use every day. *Why it matters:* familiarizes you with the exact library you'll be typing into.
3. **[Meta's Llama 3.1 model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)** — read the model card end-to-end. It's the format we'll be writing for our own release later. *Why it matters:* you learn the "shape" of what a good model card looks like, plus the specific details of our base model.
4. **[HuggingFace: "Fine-tune a pretrained model"](https://huggingface.co/docs/transformers/training)** — reading, ~1 hour. The canonical fine-tuning tutorial. Read it once even though we won't be using this exact code (we'll use TRL). *Why it matters:* familiarizes you with fine-tuning vocabulary and the shape of a training loop.
5. **[LoRA paper (skim, not deep read)](https://arxiv.org/abs/2106.09685)** — 30 min skim. Read the abstract, introduction, and Figure 1. Skip the math sections. *Why it matters:* understanding *why* LoRA works (not just that it does) is defensible in interviews.

Total reading time: ~6-8 hours. The rest of the week goes to setup and the smoke test.

---

## Concepts I'll explain in-context as we build

The following are important but you do NOT need to master them in Week 1. I'll explain each in the week where we actually use it:

- Attention mechanism (details) — Week 2
- KV cache — Week 11 (serving)
- RoPE (Rotary Position Embedding) — as-needed
- GQA (Grouped-Query Attention) — as-needed
- SwiGLU activation — not needed for our project
- DPO derivation — Week 5
- Quantization math — Week 9
- Speculative decoding — deferred to v2

Do not let unfamiliar acronyms in this list stress you. We only cover them if and when they're load-bearing for a decision we're making.

---

## Setup checklist

Do these in order. Estimated time: 2-3 hours total including account creation and troubleshooting.

- [ ] **HuggingFace account:** already exist as `centuriandip`. Log in.
- [ ] **HuggingFace API token:** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), create a token with "write" access. Save to your shell:
  ```bash
  export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXX
  ```
  Add that to `~/.zshrc` (or wherever you keep environment variables). You'll use this token in code and in `huggingface-cli login`.
- [ ] **Accept Llama 3.1 license:** visit [huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and accept the license terms. Meta requires this before you can download the model. Approval is usually near-instant to a few minutes.
      - Note: the smoke test does NOT use Llama — it uses SmolLM2-135M (non-gated, tiny, fast) — because the smoke test's job is to verify your environment, not test a specific model's access. Llama-3.1-8B is verified separately as part of environment setup.
- [ ] **Runpod account:** sign up at [runpod.io](https://runpod.io). Add payment method (add $50 to start, top up as needed — don't add $1000 upfront). Familiarize with the console but do NOT launch a pod yet.
- [ ] **Python 3.11+:** verify with `python3 --version`. If < 3.11, install via `brew install python@3.11`.
- [ ] **`uv` package manager:** faster than pip, better dependency resolution. Install via `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- [ ] **Clone the repo skeleton:**
  ```bash
  cd ~/Documents/llama-tools
  git init
  git remote add origin git@github.com:dipak-bhujbal/llama-tools.git
  ```
  Push the initial commit once we're happy with the docs.
- [ ] **Create a Python virtual environment:**
  ```bash
  cd ~/Documents/llama-tools
  uv venv --python 3.11
  source .venv/bin/activate
  uv pip install transformers datasets torch huggingface_hub
  ```
- [ ] **Smoke test — see below.**

---

## Smoke test

The purpose: verify your local Python environment can load a small model from HuggingFace and run one inference. The actual smoke test is committed at [`smoke.py`](../../smoke.py) in the repo root — read it, then run it.

Model used: **`HuggingFaceTB/SmolLM2-135M-Instruct`** — a tiny (~135M-parameter) non-gated instruction-tuned model that downloads in seconds and runs comfortably on CPU. We deliberately do NOT use Llama here — the smoke test verifies that transformers + torch + huggingface_hub work correctly, and it shouldn't be blocked by waiting for a Meta license approval.

Run it:

```bash
make smoke     # equivalent to: .venv/bin/python smoke.py
```

First run downloads the model (~300 MB) into `~/.cache/huggingface/`. Subsequent runs use the cache. CPU generation takes 5-15 seconds.

**Success:** the script prints a coherent one-sentence explanation of a transformer.

**Common failures:**
- **`AttributeError` on tokenizer.apply_chat_template:** transformers version mismatch — reinstall with `.venv/bin/pip install -U transformers`.
- **Package import error:** you're not in the repo's virtualenv. Either activate it (`source .venv/bin/activate`) or invoke Python via `.venv/bin/python smoke.py`.
- **Any HuggingFace 401 error:** your `HF_TOKEN` isn't set — check `.env` and re-run `hf auth login --token $HF_TOKEN`.

---

## Success criteria for the week

By end of Week 1 you should be able to answer these questions cold, without notes:

1. What is fine-tuning, and how does it differ from pretraining?
2. Why are we starting from an "Instruct" model instead of a "base" model?
3. What does a HuggingFace model repo contain?
4. What is a chat template and why do models care about it?
5. What is LoRA and why is it useful?
6. Roughly, what does a transformer do?

You should also have:
- A working local inference on `Llama-3.2-1B-Instruct`.
- HuggingFace + Runpod accounts live.
- A clean repo pushed to GitHub with the docs so far.

---

## What we deliberately do NOT cover in Week 1

To reduce overwhelm, the following are explicitly deferred:

- DPO, PPO, RLHF (Week 5-6)
- Attention math (touched in Week 2 as needed)
- Distributed training (Week 4 briefly, deeper only if needed)
- Quantization internals (Week 9)
- Serving optimizations (Week 11)
- MoE, speculative decoding, prefix caching — not needed for our project

If any of those show up in something you read, note the term and move on. Depth comes when a decision requires it.
