# Week 5-6 — DPO: teaching the model what NOT to do

This doc covers Direct Preference Optimization (DPO) from scratch and connects every concept to your specific project: 10,242 preference pairs assembled from 5 rule-based perturbation types, a smoke run using `train/dpo_smoke.py` (beta=0.1, lr 5e-6, 500 pairs), and the SFT weakness that motivated building all of this. By the end you should be able to walk an interviewer through DPO — intuition, loss, reference model, failure modes — as fluently as you can walk them through your SFT run.

Prerequisites: you've completed the Week 4 SFT run. You know: eval loss, LoRA rank/alpha, gradient accumulation, bf16, train-vs-eval divergence. You have your wandb DPO smoke curves open alongside this doc.

Time budget: ~12-15 hrs across the two weeks (5 hrs reading + taking notes on this doc, 3 hrs smoke run + watching metrics, 3 hrs self-quiz, 2 hrs writing ADR-006 for the smoke).

Compute already spent: ~$4.55 (SFT) + under $2 projected for the DPO smoke. Running total well under $10 of the $1,000 budget.

---

## What you need to understand by end of Week 6

- Why SFT alone is not enough — what the en-US/pt-BR miss actually demonstrates about the gap between SFT's objective and your real goal
- What a preference pair is and how DPO learns from the contrast between chosen and rejected
- What DPO's loss function computes, and what the 0.693 indifference baseline means in practice
- What the reference model does — and why our setup gets it for free without a second 8B copy in memory
- What beta controls, and why you'll ablate it in Week 8
- What degenerate collapse looks like in DPO and why we filtered 757 pairs to prevent it
- How to read the three smoke-run go/no-go signals (rewards/accuracies, margins, loss against 0.693)

---

## 1. Why SFT alone is not enough — your 4/5 gate result as the motivating example

After 9 hours and 1,095 steps, your SFT run finished with eval_loss 0.2117, zero overfitting in 11 eval checkpoints, and a qualitative gate that passed 4/5 samples correctly. The one miss: a user asked for Brazilian health news, and the model emitted `"lr": "en-US"` instead of `"pt-BR"`. Right tool. Right schema. Right argument name. Wrong value.

That miss is not a training-data gap. Your SFT dataset included xLAM examples with locale arguments. The model has seen `pt-BR` during training. What SFT failed to teach is that `en-US` in that context is **wrong** — specifically, discriminably worse than `pt-BR` — because SFT's objective has no concept of "wrong." It only has a concept of "target."

Here is the precise problem: SFT minimizes cross-entropy loss on the target token sequence. It rewards the model for getting each token right. It does not penalize the model for tokens it might have chosen instead. When the model produces `en-US`, it is making a locally plausible prediction — `en-US` is a valid BCP-47 locale tag, probably appearing near locale arguments throughout the training data, in positions structurally similar to where `pt-BR` should appear. SFT gives the model no signal that this substitution is unacceptable.

Put another way: SFT teaches the model **the target**. DPO teaches the model **the contrast between right and plausible-wrong**. The `wrong_arg_value` perturbation class in your preference data — which accounts for 1,654 of your 10,242 pairs — exists specifically to close this gap. Each pair shows the model: here is the correct locale (or argument value), and here is a plausible-but-wrong alternative. The model is trained to increase its probability of the correct one relative to the wrong one.

One more way to see the limitation: your SFT token accuracy at the end was 94.45% on the eval set. That sounds high. But function-calling is pass/fail in production — a 5.55% per-token error rate compounds across a tool call with 15-20 tokens in the arguments object. A single wrong token (a locale, an argument name, a numeric value) produces a tool call that either fails validation or returns wrong results. SFT's metric isn't aligned to the actual success criterion.

DPO's metric is: can the model rank its own correct output above a plausible incorrect one? That is much closer to what you actually care about.

---

## 2. DPO intuition, then the loss

### Preference pairs: the data shape

Every DPO training example is a triple: a prompt, a chosen response (the correct tool call), and a rejected response (a plausible but wrong tool call). In your dataset, the prompt is a rendered chat template with a system message listing the available tools and a user query. The chosen response is the exact tool call from the original xLAM example. The rejected response is that same tool call after one of your five rule-based perturbations has been applied.

An example triple, schematically:

- **Prompt:** system tools list + "Get the latest health news from Brazil"
- **Chosen:** `{"name": "get_news", "arguments": {"topic": "health", "country": "BR", "lr": "pt-BR"}}`
- **Rejected:** `{"name": "get_news", "arguments": {"topic": "health", "country": "BR", "lr": "en-US"}}`

The contrast is the entire training signal. DPO has no notion of an absolute "correct" score; it only learns from the relative gap between chosen and rejected.

### The implicit reward

RLHF — the framework DPO descends from — normally trains an explicit reward model: you collect human preference labels on pairs of completions, train a separate neural network to score completions, and then use RL to push the policy model toward high-reward outputs. This is expensive (separate training run for the reward model) and unstable (RL training can diverge or collapse).

DPO's insight, from Rafailov et al. (2023), is that under a KL-regularized RLHF objective the optimal policy has a closed-form relationship to the reward function. You can rearrange that relationship to express the reward directly in terms of the ratio of probabilities: how much more likely does the policy model think this completion is, relative to how likely the reference model thinks it is?

That ratio — specifically its log — is the **implicit reward** in DPO. You do not need to train a separate reward model. The reward is computed on-the-fly from the ratio of two probability estimates. The policy model provides one. The reference model provides the other.

Formally, for a completion y given a prompt x:

```
r(x, y) ≈ β × log(π_policy(y|x) / π_ref(y|x))
```

where β is the temperature parameter you set to 0.1 in your smoke run.

### The loss function

DPO's loss maximizes the probability that the model assigns higher implicit reward to the chosen response than the rejected one. It uses a sigmoid (the same "logistic" function from logistic regression) to convert a reward margin into a probability, then maximizes the log of that probability:

```
L_DPO = -E[ log σ( β × log(π_policy(y_w|x) / π_ref(y_w|x))
                   - β × log(π_policy(y_l|x) / π_ref(y_l|x)) ) ]
```

where y_w is the chosen (winner) response and y_l is the rejected (loser) response.

Breaking this into plain English:
1. For each pair, compute how much more (or less) likely the policy finds the chosen response compared to the reference model. That is the chosen log-ratio.
2. Do the same for the rejected response. That is the rejected log-ratio.
3. The reward margin is: chosen log-ratio minus rejected log-ratio, scaled by beta.
4. Pass that margin through a sigmoid. If the margin is large and positive (the policy strongly prefers chosen over rejected, more than the reference did), the sigmoid is close to 1.0 and the loss is close to 0.
5. Take the negative log. Minimizing loss means maximizing the margin.

The gradient of this loss simultaneously pushes two things: the policy increases its probability on the chosen response and decreases its probability on the rejected response. Both happen in a single update.

### Why 0.693 is the indifference baseline

When the reward margin is zero — the policy thinks chosen and rejected are equally good relative to the reference — the sigmoid gets zero as input. sigmoid(0) = 0.5. -log(0.5) = log(2) ≈ 0.693.

So 0.693 is the DPO loss you would see if the model had learned absolutely nothing: it assigns equal probability to chosen and rejected, same as the reference. An untrained DPO loop should start near or above 0.693. As training improves, loss drops below 0.693 — the model is beginning to prefer chosen over rejected. A loss that stays at or above 0.693 throughout training is a red flag: the model is not learning the preference signal.

In your smoke run (500 pairs, 1 epoch, ~31 steps), you want to see the loss cross below 0.693 before training ends. A final loss in the 0.5-0.65 range for a smoke run is healthy.

### Derivation sketch (interview depth)

You do not need to re-derive this cold in an interview. What you need is a convincing narrative. Here is the one to practice:

"DPO starts from the same place as RLHF: you want to maximize expected reward while staying close to the reference policy — that closeness is enforced by a KL divergence term, which prevents reward hacking. The objective is E[r(x,y)] - β × KL(π || π_ref). Rafailov et al. showed that the optimal policy for this objective has a closed-form: π*(y|x) = π_ref(y|x) × exp(r(x,y) / β) / Z(x), where Z is a normalizing constant. If you solve that for r(x,y) and substitute back into a Bradley-Terry preference model — which says P(y_w preferred over y_l) = σ(r(y_w) - r(y_l)) — you get a loss that depends only on log-probability ratios, not on r or Z directly. The reward model is eliminated. All you need are the policy model and the reference model to compute log probabilities."

That is enough for a senior AI TPM interview. If pressed for more math, say: "The key move is substituting the optimal-policy expression into the preference model, which factors out Z and leaves just the log-ratios. The loss is then a standard binary cross-entropy on which of the two log-ratios is higher."

---

## 3. The reference model: why it exists and why ours is free

### Why DPO needs a reference model at all

Without the reference model, DPO's loss would push the policy to maximize log(π_policy(y_w|x)) - log(π_policy(y_l|x)) — the raw difference in log-probabilities for chosen versus rejected. The model could satisfy this loss trivially by collapsing: assign probability 1.0 to chosen, probability 0.0 to everything else. On the training data, the loss would be perfect. In deployment, the model would produce garbage on any input that differs from its training distribution.

The reference model is the KL anchor. The loss does not just measure "does the policy prefer chosen?" — it measures "does the policy prefer chosen **more than the reference did**, while not departing too far from the reference everywhere else?" This prevents the model from optimizing away all the capabilities it learned during SFT just to maximize the preference signal on the DPO training pairs.

In program management terms: the reference model is the baseline you agreed to maintain. You are allowed to improve tool-call preference accuracy, but not at the cost of regressing the general language and reasoning capabilities the model already has. The KL term (controlled by beta) is the change-management constraint.

The reference model in your setup is specifically the SFT checkpoint — the model after Week 4, before DPO. This is the standard choice. You want DPO to refine SFT-learned behavior, not depart from it entirely. If you used the raw base model (pre-SFT) as the reference, the KL constraint would be anchored to a model that doesn't know your tool-calling format at all — the constraint would be too loose on things that matter.

### Why ours is free: adapter-disabled reference

Loading two 8B models simultaneously — one as the policy, one as the reference — requires roughly 32 GB just for weights (2 × 16 GB in bf16), before activations, gradients, or optimizer states. That does not fit on a 48 GB A6000.

Your `dpo_smoke.py` uses TRL's PEFT-adapter pattern instead. When you pass `ref_model=None` and the policy is a `PeftModel`, TRL computes reference forward passes by disabling the LoRA adapter on the same model — setting adapter weights to zero, which is mathematically equivalent to using the base model without the adapter. The reference forward pass runs on the same parameter memory as the policy forward pass; you pay no additional memory for the reference model.

In your specific case, you loaded the Week 4 SFT adapter (`outputs/sft-full`) with `is_trainable=True`, then passed `ref_model=None` to `DPOTrainer`. Under the hood, TRL calls `model.disable_adapter()` for the reference forward pass and `model.enable_adapter()` for the policy forward pass, within the same batch step. One 8B model. Two forward-pass "modes." Zero second copy.

This is the difference between your $2 smoke run and a $20 smoke run that would require a second GPU. Worth being able to explain precisely: "We use TRL's PEFT reference pattern. The reference is the adapter-disabled base, which is the same weights as the base model — the LoRA deltas are zeroed out. This means the KL constraint is anchored to the pre-adapter base, which in our case is effectively the post-SFT weights since the adapter is the SFT delta. No second model in memory."

---

## 4. Beta: what 0.1 means and why it matters

Beta (β) is the single most important DPO hyperparameter. It controls how tightly the KL constraint is applied — how close the trained policy is required to stay to the reference model at each update.

Higher beta = stronger KL constraint = slower, more conservative learning. The policy cannot move far from the reference in a single step. More stable training, lower risk of collapse, but smaller preference-accuracy improvements per epoch.

Lower beta = weaker KL constraint = the policy is free to move aggressively away from the reference to maximize preference margin. Faster apparent improvement on reward metrics, but higher risk of degenerate collapse and capability regression.

Your smoke run uses beta=0.1, which is TRL's default and the standard starting value across the DPO literature. At beta=0.1, the KL term is relatively loose — the model is allowed to learn aggressively from the preference signal. This is appropriate for a smoke run where you are verifying the pipeline works, not optimizing final quality.

For the Week 6 full run and Week 8 ablation, you will test beta ∈ {0.1, 0.3, 0.5}:

- **beta=0.1** (your smoke): loose leash. Fastest learning, highest variance, highest collapse risk.
- **beta=0.3**: moderate. This is often where the best quality/stability tradeoff lands on typical instruction-following datasets. The reference constraint is meaningfully tighter.
- **beta=0.5**: tight leash. Conservative updates. More similar to the reference in output distribution. Lower preference-accuracy gains but lower risk of capability regression.

The practical intuition: if your Week 6 reward accuracy tops out at 60% and the margins are flat, beta is probably too high — relax it. If you see reward accuracy over 90% but the model starts producing shorter, repetitive outputs on held-out prompts, beta is too low — tighten it.

Interview one-liner: "Beta is the KL-regularization temperature in DPO. Lower beta gives the model more freedom to learn preferences aggressively; higher beta keeps it closer to the reference model's distribution. We started at 0.1 and will ablate across 0.1/0.3/0.5 in Week 8 to find the quality-stability sweet spot for our specific preference data."

---

## 5. Failure modes to watch in your smoke run

### Degenerate collapse: loss near zero, everything looks too good

The most alarming DPO failure is degenerate collapse: the loss drops to ~0 within the first few steps and stays there. Reward accuracy reads ~1.0. Margins are very large. This looks like perfect learning. It is not.

Collapse means the model has found a way to trivially satisfy the loss without actually learning the preference. The most common cause on synthetic preference data is **data leakage** — the policy model has already seen essentially the same examples during SFT, so it already assigns high probability to chosen and low probability to rejected before DPO starts. The DPO loss is near-zero from step one because there is nothing to learn.

In your case, you trained SFT on the same xLAM examples that generated your preference pairs. The chosen responses in DPO are the same correct tool calls the SFT model was trained on. If the SFT model already assigns near-probability-1 to those sequences, the DPO loss will be very small from the start — not zero (because rejected still has some probability), but small.

This is one reason you filtered 757 trivial pairs before training. The ADR-005 rationale is directly relevant here: pairs where chosen and rejected differ by only one character or by whitespace carry essentially no learning signal — the model already assigns them similar probabilities, and the DPO margin is near zero. Including them wastes batch capacity and drags loss toward a misleadingly low number. After filtering, your 10,242 remaining pairs have genuine semantic contrast: wrong tool name vs. correct tool name, missing required argument vs. complete argument set, unparseable JSON vs. valid JSON. These are pairs where the SFT model's probability estimates for chosen vs. rejected will differ meaningfully, giving DPO a real gradient to work with.

If your smoke run shows immediate collapse (loss < 0.1 from step 1), check: are the 500 sampled pairs all from the same perturbation type? Are the rejected responses actually different from chosen (print a few)?

### Length/verbosity bias: margins growing but outputs getting longer

DPO has a well-documented tendency toward length bias. The log-probability of a sequence is the sum of log-probabilities of its individual tokens. A longer sequence of plausible tokens can achieve a higher total log-probability than a shorter correct sequence, even if the longer sequence contains hallucinated content. If your preference pairs have systematically different lengths between chosen and rejected — and they might, because `malformed_json` rejected responses are often slightly shorter — DPO can learn to prefer longer outputs as a proxy for "chosen," rather than learning the actual quality signal.

For function-calling, the practical symptom is the model generating extra commentary or padding around the tool call JSON when it shouldn't. Watch your qualitative samples after the smoke run. If the model output is `Here's the tool call: {"name": ...}` when gold is `{"name": ...}`, that is a verbosity artifact.

In your smoke data this risk is moderate, not high — your chosen and rejected responses are structurally similar (same JSON shape, differing in one element), so the length distribution is tight. Still, check the margin histograms if wandb shows them.

### Margins growing while accuracy is flat

A healthy DPO training curve shows reward margins (the average gap between chosen and rejected log-probability ratios) growing over time, and reward accuracy (the fraction of pairs where the policy correctly ranks chosen above rejected) also growing. These should move together.

If margins are growing but accuracy is flat near 0.5, the model is becoming more confident in its predictions without improving the predictions themselves. This can indicate your preference pairs are low-signal — the rejected examples are not distinct enough from chosen for the model to reliably discriminate. The fix is to examine which perturbation types are driving this: likely `wrong_arg_value` near-misses where the value difference is subtle.

If accuracy is growing but margins are flat, the model is ranking correctly but with low confidence. This can be fine at the smoke stage — margins often grow more slowly than accuracy early in training.

---

## 6. The three go/no-go smoke signals

Your smoke run covers 500 pairs, 1 epoch, ~31 optimizer steps. This is too small to produce a production-quality adapter; that is not the goal. The goal is to verify the training loop is working correctly before you commit to the Week 6 full run (10,242 pairs, 3 epochs, ~$20-30). These three signals are your go/no-go criteria.

### Signal 1: rewards/accuracies > 0.5 and trending upward

TRL's DPO trainer logs two metrics every `logging_steps` steps: `rewards/chosen` and `rewards/rejected` (the average implicit reward the policy assigns to each side, relative to the reference), and `rewards/accuracies` (the fraction of pairs where rewards/chosen > rewards/rejected).

What it measures: whether the model is actually learning to prefer chosen over rejected. An accuracy of 0.5 means random chance — the model cannot discriminate. An accuracy trending from 0.5 toward 0.7+ over 31 steps means the model is picking up the preference signal.

What 0.5 flat means: something is wrong. Either the loss is collapsing for a non-learning reason, or the pairs have no discriminable signal, or there is a data-format bug where chosen and rejected are being fed in the wrong order.

Target for your smoke: accuracy above 0.5 and still climbing at step 31. If it plateaus at 0.6+ by step 20, that is a strong smoke signal.

### Signal 2: margins growing (rewards/margins)

TRL also logs `rewards/margins` — the average of (rewards/chosen - rewards/rejected) per batch. This is the average margin by which the policy ranks chosen over rejected.

What it measures: whether the policy's preference is strengthening over time, not just whether it is correct. A model that assigns chosen a reward of 0.01 and rejected -0.01 technically has the right ordering, but barely. A model with chosen at 0.8 and rejected at -0.3 has learned a much sharper distinction.

For your smoke, margins starting near zero (or slightly negative, since the reference model may actually already prefer chosen slightly for easy pairs) and growing positively over 31 steps is the healthy pattern. You don't need large margins from a 500-pair, 1-epoch smoke run — you need positive and increasing.

Flat margins that never grow despite accuracy improving can indicate the model is rank-correct but not confident, which is acceptable for a smoke. Negative margins that stay negative is a problem: the model is learning to prefer the rejected response, which happens if chosen/rejected are swapped in your data-loading code.

### Signal 3: loss trending below 0.693

As described in section 2: 0.693 is the indifference baseline (log 2). A smoke run that never crosses below 0.693 means the model has not learned any preference signal. A healthy smoke finishes with loss in the 0.5-0.65 range after 31 steps on 500 pairs.

This is the coarsest of the three signals — accuracy and margins are more informative — but it is the fastest sanity check. If you look at your wandb loss curve and it is still flat at 0.70 after step 15, stop early and diagnose before completing the full epoch.

What "below 0.693 and still dropping" means: the model is learning, training is not collapsed, you have a real gradient signal from the preference pairs. Green light for the Week 6 full run.

Three signals, joint verdict: all three moving in the right direction simultaneously is a clean pass. Any one of them flat or moving wrong is a hold — diagnose before scaling to the full run.

---

## What we do NOT cover in Weeks 5-6

- ORPO and KTO (alternative preference optimization methods — acknowledged in ADR-002, deferred as future work)
- Reward model training (the explicit RLHF path we rejected in ADR-002 as compute-prohibitive)
- Online DPO / iterative DPO (generating new rejected samples from the policy mid-training)
- SimPO and IPO (DPO variants with modified loss formulations — valid research but not for this project)
- Per-perturbation-type reward accuracy breakdown (Week 6 full analysis, not smoke scope)
- Beta ablation (Week 8)

---

## Self-quiz: answer these cold

These are the questions you should be able to answer without this doc in front of you. Some are conceptual; some require you to read hypothetical numbers the same way you would in an interview.

**1.** Your SFT model passed 4/5 qualitative samples. The one failure was `"lr": "en-US"` instead of `"lr": "pt-BR"`. Explain precisely why SFT training could not have prevented this miss — what does SFT's objective actually optimize, and what does it not optimize?

**2.** A colleague says: "DPO is just fine-tuning on the chosen responses and doing negative fine-tuning on the rejected ones." Is that accurate? What is the key thing they are leaving out?

**3.** What is the implicit reward in DPO? Write it symbolically and translate each component into plain English.

**4.** If your DPO smoke run produces a loss of exactly 0.693 at every step, what does that tell you? What are the two most likely root causes?

**5.** Why does your setup use `ref_model=None` in DPOTrainer? What does TRL actually do when ref_model is None and the policy is a PeftModel? Be specific about which model weight state produces the reference forward pass.

**6.** You run your smoke and observe the following metrics at step 31 (final step):
   - `rewards/accuracies`: 0.52 (started at 0.49 at step 1)
   - `rewards/margins`: 0.08 (started at 0.01 at step 1)
   - Loss: 0.681 (started at 0.698 at step 1)
   
   Is this a go or a no-go for the Week 6 full run? Justify your answer for each signal.

**7.** Same setup, different numbers. Step 31 results:
   - `rewards/accuracies`: 0.97
   - `rewards/margins`: 2.4
   - Loss: 0.02
   
   Should you be excited? What is the specific failure mode these numbers suggest, and what would you check in the data first?

**8.** Why did you filter 757 preference pairs before DPO training? Connect the filtering decision specifically to what happens in the DPO gradient when chosen and rejected are near-identical.

**9.** You set beta=0.1 for the smoke run. Your colleague argues you should use beta=0.5 to be safer. Walk through the tradeoff: what does beta=0.5 prevent, and what does it cost?

**10.** An interviewer at a frontier lab asks: "Walk me through your full post-training pipeline from raw data to deployed model." Give a 90-second answer that covers SFT, why SFT alone wasn't sufficient, the preference data methodology (rule-based perturbations, why you chose them, how many pairs you ended up with after filtering), DPO's role, and how you validated each stage.

---

## Success criteria for Weeks 5-6

By the end of Week 6 you should:

- Be able to answer all 10 self-quiz questions cold, including questions 6 and 7 which require reading hypothetical numbers
- Have run the DPO smoke (`train/dpo_smoke.py`) and confirmed all three go/no-go signals are positive
- Know your exact smoke numbers: starting loss, final loss, final rewards/accuracies, final rewards/margins
- Have an ADR filed for the smoke run in `docs/decisions/` covering: the reference model pattern, beta choice, smoke results, and go/no-go verdict
- Be able to narrate the DPO derivation sketch from section 2 in 60-90 seconds without reading it — your own words, not a recitation

If your smoke fails any go/no-go signal, diagnose before filing the ADR: print 5 raw preference pairs to verify the data format, check that chosen and rejected are not swapped, verify the adapter loaded correctly from `outputs/sft-full`. Most smoke failures are data-format or path issues, not algorithmic problems.

Ping when you have the smoke results — we will read them together before you commit to the full Week 6 run.
