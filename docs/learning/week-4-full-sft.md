# Week 4 — Full SFT run: reading your own results

This doc covers the full LoRA-SFT run you just completed: Llama-3.1-8B-Instruct, 11,660 training examples, 3 epochs, RTX A6000 48 GB, ~8.8 hours, ~$4.40. Every concept below is grounded in numbers from that specific run. By the end you should be able to walk an interviewer through every decision cold.

Prerequisites: you've read weeks 1 and 2 and watched the training run complete. You should have your wandb loss curves open as you read this.

Time budget: ~10-12 hrs this week (4 hrs reading this doc + reviewing curves, 4 hrs on the self-quiz with notes open, 4 hrs writing your ADR for this run).

Compute already spent: ~$4.40 from the $1,000 budget.

---

## What you need to understand by end of Week 4

- What train loss and eval loss each measure, and why they must diverge somewhat
- Why you keep the epoch-2 checkpoint rather than the final one
- What LoRA rank and alpha actually do to the weight matrices, and why it's ~0.5% of parameters
- Why you need gradient accumulation to reach effective batch 32 on a single GPU
- Why bf16 and gradient checkpointing let you fit this run on 48 GB
- What DDP and FSDP are, at the level you'd need to describe them in a system-design interview

---

## 1. Train loss vs. eval loss — reading your own curves

### What each one measures

**Train loss** is the average cross-entropy loss the model produces on the batch of examples it just trained on, measured *after* the weight update. It tells you: is the model getting better at predicting the next token in the training set?

**Eval loss** is the same metric computed on 500 held-out examples the model never trained on, with weights frozen. It tells you: is the model generalizing, or just memorizing?

Think of it like a PMP exam analogy: train loss is your score on the practice questions you've already seen; eval loss is your score on the real exam with fresh questions.

### Your run's numbers

Your run: 11,660 train examples, 500 eval examples, 3 epochs, eval every 100 steps (roughly every 1/3 of an epoch). That gives you ~10 eval checkpoints to look at on the wandb curve.

Healthy pattern you should see:
- Steps 0–365 (epoch 1): train loss drops steeply from ~1.5–2.0 down toward ~0.8–1.0. Eval loss follows but lags a little — that lag is normal and expected.
- Steps 365–730 (epoch 2): both continue dropping but more gradually. The gap between train and eval loss starts widening slightly. This is the sweet spot.
- Steps 730–1,095 (epoch 3): train loss keeps falling (the model is still memorizing the training set), but eval loss may plateau or tick upward. That uptick is the early signature of overfitting.

### Why they diverge

The model has 11,660 examples to memorize and 8B parameters (of which LoRA touches ~40M). By epoch 3, the model has seen every training example three times and starts fitting the exact phrasing rather than the underlying pattern. That shows up as: train loss still falling, eval loss plateauing or rising.

Overfitting on your curves would look like a visible "V" or plateau in eval loss while the train loss curve keeps going down. If your epoch-3 eval loss is more than ~0.05–0.10 nats higher than your epoch-2 eval loss, you overfit in epoch 3.

### Why you keep the epoch-2 checkpoint

You checkpoint at every eval step (every 100 optimizer steps) and keep the checkpoint with the best eval loss. In a 3-epoch run on this dataset size, that best checkpoint almost always lands in epoch 2, not epoch 3. In program-management terms: the epoch-2 checkpoint is the version that shipped to real users; epoch 3 is a hot-fix that regressed quality because the team was optimizing against the wrong metric.

The practical implication: when you push to HuggingFace, you push the epoch-2 adapter, not `final_model/`.

---

## 2. LoRA mechanics — what r=64/alpha=128 actually does

### The core idea

Every attention layer in a transformer has weight matrices: Q (query), K (key), V (value), and O (output projection). In Llama-3.1-8B, each of these is a roughly 4096 × 4096 matrix — about 16.7M parameters per matrix, four matrices per layer, 32 layers. Full fine-tuning would update all of those: ~2.1B parameters just in attention.

LoRA's insight: you do not need to move around in the full 4096-dimensional space to adapt the model's behavior. Most of the change you want lives in a much lower-dimensional subspace. So instead of updating W directly, LoRA adds two small matrices:

```
ΔW = B × A
```

where A is (r × d_in) and B is (d_out × r). For r=64 and d=4096: A is 64 × 4096 (262,144 params) and B is 4096 × 64 (262,144 params). Total per matrix: 524,288 params instead of 16.7M. That's 3.1% the size of one full matrix.

You targeted q, k, v, and o projections across all 32 layers: 4 matrices × 32 layers × 524,288 params ≈ 67M LoRA parameters. Llama-3.1-8B has ~8B total parameters. 67M / 8,000M ≈ 0.84% — call it "under 1% of parameters" in interviews. (Your specific config may be slightly different depending on projection sizes, but ~0.5–1% is the correct order of magnitude.)

During training: A and B are updated via backprop. The original W is frozen. During inference: the update is applied as W + B×A (or the adapter can be merged permanently into W for zero-latency inference).

### What rank controls

Rank r is the width of that bottleneck. Higher rank = the adapter can represent more complex changes = more expressiveness = more parameters to train = more memory = more risk of overfitting.

r=64 is on the higher end for SFT. Common choices:
- r=8–16: cheap, enough for style/format adaptation
- r=32–64: good for task-specific knowledge adaptation (your use case)
- r=128+: approaching full fine-tuning territory in cost; rarely needed for SFT

### What alpha controls

Alpha (α=128) is a scaling factor applied to the LoRA update: the effective update is `(α/r) × B×A`. With α=128 and r=64, the scaling factor is 2.0. This means the LoRA update is applied at 2× the "natural" learning rate implied by the optimizer. The ratio α/r is what matters, not the absolute value of either. α=128, r=64 gives ratio 2.0 — same as α=64, r=32.

Interview one-liner: "Alpha scales how aggressively we apply the low-rank update. A ratio of 2 is a common default that provides a slight boost without destabilizing training."

### LoRA vs. full fine-tuning: the cost/quality tradeoff

| | LoRA (your run) | Full fine-tuning |
|---|---|---|
| Params updated | ~67M (< 1%) | ~8B (100%) |
| GPU memory for optimizer states | ~500 MB | ~96 GB |
| Training time (8.8 hrs) | feasible on 1× A6000 | needs 4× A100 80GB minimum |
| Quality delta | ~2–5% worse on domain tasks | baseline |
| Adapter file size | ~250 MB | 16 GB |

For your use case (function-calling adaptation of an already strong model), the quality delta is well within acceptable range, and LoRA is the only feasible single-GPU option.

---

## 3. Gradient accumulation + bf16 + gradient checkpointing — the memory triangle

### Why memory is the constraint

Training an 8B model requires holding in GPU memory simultaneously: the model weights (16 GB in bf16), the optimizer states (another ~32 GB for Adam's momentum and variance in fp32), the activations for backprop, and the current batch of tokens. On a 48 GB card, this is tight.

There are three dials you tuned to make it fit:

### Gradient accumulation (why effective batch = 32)

Per-device batch size of 8 means each forward/backward pass processes 8 training examples and computes gradients for them. But you set `gradient_accumulation_steps=4`, which means: accumulate gradients across 4 consecutive mini-batches before calling the optimizer step.

Effective batch size = per_device_batch × grad_accum_steps = 8 × 4 = 32.

Why does this matter? Larger effective batch = more stable gradient estimate = smoother loss curve + potentially higher LR. The cost: 4 forward/backward passes per optimizer step instead of 1. Memory cost is identical to batch size 8 because you only ever hold one mini-batch in GPU memory at once. The gradients just accumulate in-place before being applied.

Program-management analogy: instead of making a procurement decision after hearing from one stakeholder (batch 8), you poll four stakeholders and average their input before deciding (effective batch 32). Better signal, same meeting room size.

### bf16 mixed precision

bf16 (bfloat16) stores each weight as 16 bits instead of 32 bits (fp32), cutting weight memory roughly in half. "Mixed precision" means: weights and activations are stored in bf16 for the forward pass; gradients are computed in fp32 precision before being applied. The optimizer states (Adam's momentum/variance) remain in fp32.

bf16 specifically (vs fp16): bf16 has the same exponent range as fp32, so it doesn't overflow as easily. fp16 can overflow at large gradient values, requiring a "loss scaler" to prevent NaN loss. bf16 avoids this entirely, which is why it's preferred on Ampere-and-newer GPUs (A6000 = Ampere = supports bf16 natively).

Memory saved on your run: ~8 GB compared to fp32 weights — that headroom is what makes the A6000 viable.

### Gradient checkpointing

During the backward pass, you normally need to store all intermediate activations from the forward pass so you can compute gradients. For a 2,048-token sequence through a 32-layer transformer, those activations are enormous.

Gradient checkpointing trades compute for memory: instead of storing all activations, it stores only activations at certain "checkpoint" boundaries and recomputes the rest during the backward pass. The cost is roughly +30% compute time (you recompute some forward passes twice). The benefit is that activation memory drops from O(sequence_length × layers) to O(sequence_length × sqrt(layers)).

In practice on your run: gradient checkpointing probably saves 8–12 GB of activation memory, which is what makes max_seq_len=2048 feasible. Without it, you would have had to drop to 512 or 1024 tokens.

### The triangle summary

| Technique | What it saves | What it costs |
|---|---|---|
| Gradient accumulation | Nothing directly — enables larger effective batch | +compute (minor) |
| bf16 | ~8 GB weight memory | Slight precision loss (negligible in practice) |
| Gradient checkpointing | ~8–12 GB activation memory | +~30% compute time |

Combined: three complementary tools that make a 48 GB card viable for 8B-param training. Each one alone is insufficient; together they're why your $4.40 run was possible.

---

## 4. Distributed training — DDP vs. FSDP (interview level only)

Your Week 4 run was single-GPU. You do not need to operate distributed training. But interviewers at frontier labs will ask about it, and you need to explain the tradeoff clearly.

### Data Parallel training (DDP — DistributedDataParallel)

DDP is the simpler approach. Each GPU holds a full copy of the model. Training data is split across GPUs. Each GPU does its own forward/backward pass on its slice of the batch. After each step, GPUs synchronize: they all-reduce (average) their gradients across the ring so every GPU applies the same gradient update.

Result: effective batch size scales linearly with GPU count. Communication cost is one all-reduce per step.

Limitation: every GPU must hold the full model in memory. For 8B params in bf16, that's ~16 GB per GPU just for weights, plus optimizer states. This is the bottleneck — DDP doesn't help you fit a model that exceeds a single GPU's memory.

### Fully Sharded Data Parallel (FSDP)

FSDP shards everything — weights, gradients, optimizer states — across GPUs. No single GPU holds a complete copy of any layer's parameters. When a layer is needed for the forward pass, GPUs gather the shards on-the-fly, compute, then immediately discard the gathered parameters.

Result: each GPU's memory requirement drops proportionally to N_gpus. A model that requires 4× a single GPU's memory can run on 4 GPUs with FSDP.

Cost: more communication overhead than DDP because you gather shards before every layer (not just once per step). On high-bandwidth interconnects (NVLink inside a server), this is fast. Over slow interconnects (consumer PCIe or across nodes), FSDP can become communication-bound.

### When you'd use each

| | DDP | FSDP |
|---|---|---|
| Model fits on one GPU | Yes | Yes (but overkill) |
| Model doesn't fit on one GPU | No | Yes |
| Batch size too small per GPU | Helps (more GPUs = bigger batch) | Helps |
| Communication overhead | Low (one all-reduce/step) | Higher (gather/scatter per layer) |
| Code complexity | Simple | Complex |

**Interview one-liner for your situation:** "Our Week 4 run fit on a single A6000 48 GB using LoRA + bf16 + gradient checkpointing. For a full-parameter fine-tune of 8B or larger, or for 70B+ LoRA, we'd move to FSDP because DDP would OOM on any single GPU."

---

## What we do NOT cover in Week 4

- Multi-node FSDP (across multiple machines — relevant at 70B+, not for our project)
- Pipeline parallelism (splitting layers across GPUs — even more exotic)
- Tensor parallelism (splitting individual matrix multiplications — frontier-lab territory)
- DPO or preference optimization (Week 5–6)
- Quantization-aware training (Week 9)

---

## Self-quiz: answer these cold after your run completes

These are the questions you should be able to answer in a 45-minute technical interview without notes. For the ones marked "open your wandb," have the curve in front of you the first time but then practice closing the tab.

**1.** What does cross-entropy loss measure in this context? Why is lower better?

**2.** (Open your wandb) What was your train loss at step 100? At step 1,095 (final step)? What was your best eval loss, and at which step did it occur?

**3.** Why did you evaluate every 100 steps rather than only at the end of each epoch? What would you miss if you only evaluated at epoch boundaries?

**4.** Your eval loss is higher than your train loss throughout training. Is that a problem? Why or why not?

**5.** Explain LoRA to someone who has never heard of it, using only an analogy from project management or engineering. No equations.

**6.** Your run used r=64, alpha=128. What would likely happen if you re-ran with r=8, alpha=16, all else equal? What about r=128, alpha=256?

**7.** You used gradient accumulation steps of 4 with per-device batch size of 8. What is the effective batch size? If you had a second GPU and removed gradient accumulation, what per-device batch size would you need to keep the effective batch size the same?

**8.** Why did you use bf16 instead of fp32? What property of bf16 makes it safer than fp16 for this use case?

**9.** Gradient checkpointing added ~30% to your compute time. Your run took 8.8 hours. Roughly how long would the same run take without gradient checkpointing, assuming you could fit it in memory? What would you have to give up in your config to run without gradient checkpointing on the same GPU?

**10.** A recruiter at a frontier lab asks: "How would you scale this to Llama-3.1-70B?" Walk through what changes and why, at the infrastructure level.

---

## Success criteria for the week

By end of Week 4 you should be able to answer all 10 quiz questions cold. You should also:

- Have the epoch-2 checkpoint pushed to HuggingFace as a private repo
- Have an ADR written in `docs/decisions/` covering your config choices and what the loss curves showed
- Know your exact numbers: best eval loss, the step it occurred, final train loss, total cost
- Be able to explain the difference between DDP and FSDP in under 90 seconds to a non-technical interviewer

Ping me when you're ready to write the ADR together, or if your loss curves show something unexpected (eval loss rising before epoch 2, NaN loss, or a completely flat train loss curve in epoch 1).
