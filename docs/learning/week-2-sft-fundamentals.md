# Week 2 — SFT fundamentals + first cloud training run

This is your study ramp for Week 2. You've completed Week 1 fundamentals, so this doc assumes you know: what a transformer is, what fine-tuning is, what LoRA is (roughly), how HF's `transformers` library works, and how to load a model locally.

Week 2 does three things: (1) go deep enough on SFT to defend the choice in an interview, (2) get a rented GPU running and prove the training loop works end-to-end, (3) push your first fine-tuned model artifact to HuggingFace.

Time budget: ~15-20 hrs across the week (roughly 4 hrs learning + 4 hrs Runpod/wandb setup + 4 hrs the actual smoke run + 4 hrs debugging and buffer).

Compute budget for Week 2: ~$15-25 (per `PLAN.md`).

---

## What you need to understand by end of Week 2

By end of week, you should be able to explain these in your own words to a colleague:

- **What SFT does mathematically.** Given input–target pairs, minimize cross-entropy loss on the target tokens. That's it. Every fancy training technique reduces to "compute loss, backprop gradients, update weights."
- **The training loop.** Batch → forward pass → loss → backward pass → optimizer step → repeat. What each step actually does in memory.
- **LoRA in more depth.** Not just "small adapter" — but: which weight matrices are targeted (typically q_proj and v_proj), what rank and alpha do, why LoRA works despite training so few parameters, and roughly how much memory/compute you save.
- **Reading training curves.** What a healthy train-loss curve looks like (steady exponential decay). What overfitting looks like (train drops, eval flatlines or rises). What divergence looks like (loss goes to inf or nan).
- **Chat templates during training.** Why formatting matters. What happens if you train on unformatted conversation data (the model learns to generate raw text without the special tokens, which breaks inference).
- **TRL SFTTrainer.** What arguments matter, what defaults to trust, what to configure explicitly (LoRA config, batch size, learning rate, sequence length, epochs).

---

## Recommended reading order (all free)

1. **[Sebastian Raschka: "Practical Tips for Finetuning LLMs Using LoRA"](https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html)** — 45-min read. Best plain-English explanation of what LoRA actually does and why the hyperparameters matter. *Why it matters:* you'll be tuning LoRA hyperparameters starting Week 4; this article makes the choices intuitive rather than magical.

2. **[HuggingFace TRL: SFTTrainer docs](https://huggingface.co/docs/trl/main/en/sft_trainer)** — 30 min. Read the "Quickstart" and "Advanced Usage" sections. *Why it matters:* SFTTrainer is what your Week 4 script calls; you should recognize every argument.

3. **[LoRA paper — Section 4 (Experiments)](https://arxiv.org/abs/2106.09685)** — 30 min focused read. You skimmed this in Week 1; this time read Section 4 carefully to see how the authors chose ranks and target modules empirically. *Why it matters:* justifies our starting choice (rank 64, alpha 128) and gives you defensible references.

4. **[HuggingFace NLP Course Chapter 3](https://huggingface.co/learn/nlp-course/chapter3)** — 1-2 hours. The canonical fine-tuning tutorial. You skimmed in Week 1; now do a careful read with the Trainer walkthrough. *Why it matters:* you'll see the exact patterns TRL builds on top of.

5. **[wandb quickstart](https://docs.wandb.ai/quickstart)** — 20 min. *Why it matters:* every training run should log to wandb so you can debug loss curves and compare runs. Not optional for serious training work.

**Optional deeper dives (do only if you finish everything else early):**
- [QLoRA paper](https://arxiv.org/abs/2305.14314) — LoRA + 4-bit quantization for even cheaper training. We may adopt this in later weeks if budget gets tight.
- [Chinchilla paper](https://arxiv.org/abs/2203.15556) — compute-optimal training. Not needed for fine-tuning specifically but shapes intuition about what compute buys you.

---

## Concepts I'll explain in-context as we build

You don't need to master these upfront — they come up as we hit them in code:

- **Attention head computation math** — only if we need to debug attention behavior (unlikely)
- **Optimizer choice** (AdamW vs Lion) — we default to AdamW, defer alternatives
- **Learning rate schedules** (cosine vs linear vs constant) — we use cosine with warmup; I'll explain when we set it
- **Gradient checkpointing** — memory optimization, comes up if we hit OOM
- **Distributed training details** (DDP vs FSDP) — Week 4 uses single-GPU so this stays theoretical for now

---

## Setup checklist

Do these in order. Estimated time: 3-4 hours including debugging.

- [ ] **Install train dependencies:**
  ```bash
  cd ~/Documents/llama-tools
  make install-train
  ```
  This pulls trl, peft, accelerate, bitsandbytes, wandb into your existing venv.

- [ ] **wandb account:** sign up at [wandb.ai](https://wandb.ai). Free tier is fine. Get your API key from settings.

- [ ] **Save wandb API key to `.env`:**
  ```
  WANDB_API_KEY=<your-key>
  ```
  Add it right below the existing `HF_TOKEN` entry.

- [ ] **Runpod: launch your first pod.**
  - Log in at [runpod.io](https://runpod.io).
  - Templates → search "PyTorch 2.4" → pick the community `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` image (or newer if available).
  - GPU: **1x A100 40GB** in the **Community Cloud → Spot** tier (~$0.44/hr).
  - Container disk: 40 GB (default is fine; we need room for the 8B model download).
  - Volume: 10 GB (persists between pod stops if you want to save weights between sessions — skip for smoke test).
  - Deploy. Wait ~1-2 min for it to boot.

- [ ] **Connect to your Runpod pod.**
  - Easiest: click "Connect" → "Start Web Terminal" for a browser shell.
  - Better: click "Connect" → SSH details, then `ssh -i ~/.ssh/id_ed25519 root@<pod-ip> -p <port>` from your Mac terminal. (Requires you to add your SSH pub key in Runpod settings first.)

- [ ] **Set up the pod:**
  ```bash
  # Inside the Runpod pod
  cd /workspace
  git clone https://github.com/dipak-bhujbal/llama-tools.git
  cd llama-tools
  pip install --upgrade pip
  pip install -e ".[train]"
  export HF_TOKEN=<paste your token>
  export WANDB_API_KEY=<paste your key>
  hf auth login --token $HF_TOKEN
  wandb login $WANDB_API_KEY
  ```

- [ ] **Verify GPU visibility:**
  ```bash
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
  ```
  Should print `CUDA available: True, GPU: NVIDIA A100-SXM4-40GB` (or similar).

- [ ] **Run the Week 2 smoke test.** (Script committed in the next section — I'll add it to the repo before you start Week 2.)

- [ ] **Watch the training run in wandb.** You should see a loss curve appearing in real-time.

- [ ] **After training completes, push the LoRA adapter to HF as a private repo.**

- [ ] **STOP THE POD when done.** Runpod bills per second the pod is running, not per compute used. Forgetting to stop a pod for 24 hours costs ~$10.

---

## The Week 2 smoke test

**Goal:** LoRA-SFT of Llama-3.1-8B-Instruct on a tiny slice (500 examples) of Glaive function-calling data. Should complete in ~30-60 minutes on 1x A100 40GB. Cost: ~$0.25-0.50.

**Success criteria:**
- Training loss curve decreases smoothly (not flat, not erratic, not exploding)
- Eval loss (on 50 held-out examples) also decreases, roughly tracking train loss
- The final LoRA adapter loads cleanly and can be applied to the base model
- Adapter pushed to HF as `centuriandip/llama-3.1-8b-tools-week2-smoke` (private for now)

**Failure modes and what they mean:**
- **Loss immediately goes to NaN or inf:** learning rate is too high, or LoRA config is broken. Fix: drop LR, verify LoRA target modules match Llama's actual attention module names.
- **Loss doesn't decrease at all:** LoRA rank too small, or you're training on wrong data. Fix: bump rank, verify data format with a print statement.
- **Loss decreases but eval loss diverges immediately:** you're overfitting a tiny dataset. This is expected on 500 examples; not a concern for smoke test.
- **OOM (Out of Memory) errors:** batch size too large. Fix: drop batch size to 4 or 2, enable gradient checkpointing.
- **CUDA errors:** wrong CUDA version between torch and the Runpod image. Fix: use the recommended Runpod image, don't override torch version.

I'll drop the actual `train/sft_smoke.py` script and a walkthrough into the repo *before* Week 2 begins — but I'll write it *with* you (Mode A), not before. When you're ready to start Week 2, ping me and we'll write the script together, explaining each argument as it lands.

---

## What we do NOT cover in Week 2

- Full-parameter fine-tuning (LoRA only)
- Multi-GPU / distributed training (Week 4 stays single-GPU)
- DPO or any preference optimization (Week 5-6)
- Advanced learning rate schedules beyond cosine-with-warmup
- Custom loss functions
- Reward modeling
- Model merging techniques beyond simple LoRA merge

Any unfamiliar term: check `glossary.md`. If not there and the term is worth knowing, ping me to add it.

---

## Success criteria for the week

By end of Week 2 you should be able to answer cold, without notes:

1. What does SFT minimize? (Cross-entropy loss on target tokens.)
2. Why do we use LoRA instead of full fine-tuning? (Cost, memory, speed; quality is nearly identical for our use case.)
3. What determines whether a LoRA fine-tune has capacity to learn a task? (Primarily LoRA rank; secondarily the target modules.)
4. What's a healthy training loss curve? (Steady exponential decay, no explosions, eval loss tracks train loss loosely.)
5. What arguments to SFTTrainer matter most? (Model, dataset, per_device_train_batch_size, gradient_accumulation_steps, learning_rate, num_train_epochs, LoRA config via peft_config.)
6. How much did Week 2's smoke test cost? (~$0.25-0.50 — should be exact from Runpod usage stats.)

You should also have:
- Working Runpod workflow (launch → SSH → train → push → stop)
- Working wandb account with your first run logged
- A private HF repo containing your first fine-tuned LoRA adapter
- Roughly $15-25 spent from the total $1000 budget

Ping me when you're ready to start Week 2 setup. We write `sft_smoke.py` together.
