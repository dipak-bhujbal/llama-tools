# Postmortem — §0 qualification probe attempt, 2026-08-08

**Author:** Claude, written independently at the owner's request (#general msg 2828).
Codex is writing its own separately; where they differ, neither is authoritative
until reconciled.

**Outcome: the probe produced ZERO generations.** There is no §0 result.

---

## 1. What was attempted

A two-category §0 endpoint qualification probe at reviewed commit
`2d8abdbc85ba24f49d833b83dc1d6b53712fe6e1`: `multiple` (200 prompts) then
`simple_python` (400), each scored on **base** and **shipped-SFT** candidates —
1,200 generations, sharing one absolute deadline.

Owner-approved planning estimate **$0.19–$0.39** against a **$0.45** ceiling
(#general 2717, 2719, 2720). Those are planning figures. **No measured charge is
reported anywhere in this document.**

## 2. Timeline (UTC)

| time | event |
|---|---|
| 01:43:37 | Pod created — RTX 4090 24 GB, community cloud, `costPerHr 0.34`, provider auto-termination set **at creation** for 02:53:37 |
| 01:44:18 | Owner assumes sole-operator role — **41 s after creation**; elects to keep the running pod |
| ~01:45–01:57 | Bootstrap: bundle sha256 verified, checkout at the reviewed commit, clean tree asserted, venv, deps, 16 GB weights, preflights — **all passed** |
| 02:00:25 | Paid generation begins: `multiple`, with **3021 s** of the shared deadline remaining |
| 02:00:25–02:01:29 | Tokenizer loads; 291 base shards load; **SFT adapter attaches successfully** |
| **02:01:29** | **CRASH** on the **first prompt** |
| 02:01:48 | Launcher exits **68** after 83 s. `simple_python` never starts |
| 02:01:48–02:35:54 | **~34 minutes with the pod allocated and nothing running** |
| ~02:36 | Final pull, provider snapshot, owner deletes pod; `runpodctl pod list` → `[]` |

## 3. The failure

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
  modeling_llama.py:267  apply_rotary_pos_emb(...)
  modeling_llama.py:166  q_embed = (q * cos) + (rotate_half(q) * sin)
  modeling_llama.py:142  torch.cat((-x2, x1), dim=-1)      ← surfaced here
```

Manifest: `status=incomplete`, `rows_written=0`, elapsed `64.0178s`.

**CUDA reports these errors asynchronously.** The traceback shows where the
error *surfaced*, not where it *originated*. Layer 0's rotary embedding is a
synchronization point, not an accusation.

## 4. Cause: indeterminate — and why that is the right word

Not "unknown because we stopped looking." **The run did not retain the evidence
required to distinguish the candidates.**

**Excluded, with evidence:**

| candidate | evidence |
|---|---|
| credentials / model gating | SFT adapter attached successfully |
| corrupt fixtures | every fixture verified by sha256 **and** revision before the paid command |
| extreme context | first prompt **609 tokens**; full range 280–999 (measured by codex on the pinned tokenizer) |
| unsupported library versions | Torch 2.9.1 within Transformers 5.14.1's advertised PyTorch range |
| below-minimum driver | 570.133.20 meets CUDA 12.8.1's published minimum |
| FlashAttention / xformers | not installed |

**Still open — could not be separated:** consumer-GPU/community-node
instability · a CUDA or kernel bug · the **PEFT wrapper path** (the "base"
candidate also runs through `PeftModel` under `disable_adapter()`; **raw
unwrapped base was never exercised**) · `device_map="auto"` placement ·
memory pressure (**peak/free VRAM was never captured**).

**A dead end recorded so it is not re-derived:** I proposed that §3.4's
`torch 2.13.0` pin implied incompatibility. **Wrong, and out of scope** —
`docs/prereg-study2.md:1863–1868` states §3.4 governs *training arms* while
`requirements-probe.txt` governs §0 and deliberately leaves Torch unpinned.
Selecting 2.13 for §0 would **cross a preregistration boundary**, not restore a
pin. Codex caught it.

## 5. What actually went wrong — process, not silicon

Four defects. **All four are mine**, and three share one shape: *a failure that
was hidden rather than surfaced.*

1. **No exact-production smoke.** Bootstrap asserted imports and
   `torch.cuda.is_available()`. Neither predicts a kernel fault.
   `requirements-probe.txt:12–19` *anticipates this failure mode in a comment* —
   **a comment is not a check.** One real generation would have failed in
   seconds instead of consuming a pod.
2. **No liveness monitoring.** The snapshot loop I designed copied a **dead
   run's** log every two minutes while I told the owner silence was normal.
   It guarded against data loss and never against death. **~34 minutes.**
3. **Insufficient failure telemetry.** No `hf_device_map`, no peak/free VRAM, no
   resolved attention path, no GPU/Xid health. These are precisely the
   discriminators §4 now lacks — which is why the cause is indeterminate.
4. **Instruction-induced secret exposure.** My operator sheet contained
   `export HF_TOKEN="<TOKEN>"` inside a block the owner was asked to run **and
   paste back**. The echo carries the secret. The owner did nothing wrong; the
   instruction was defective. Correct form: `read -rsp 'token: ' HF_TOKEN; echo`.

**Also mine:** `git clone … 2>/dev/null` in a helper step suppressed
`remote HEAD refers to nonexistent ref` (a single-branch bundle has no HEAD),
producing an empty directory and a confusing missing-file error one step later.
Same shape as defect 1.

## 6. What worked

- Fail-closed design **refused to spend on generation after the fault**;
  `simple_python` never started and nothing auto-retried.
- The **shared absolute deadline** behaved correctly. The earlier 50/50 split
  would have starved `simple_python` (800 generations vs 400) — visible in the
  live log as `3021s` handed to the first command rather than half of it.
- Provider auto-termination was set **at creation**, where it cannot be retrofitted.
- The owner's snapshot loop **preserved the only evidence we have.**
- Partial-run finalizer refused to let deletion happen before artifacts and the
  provider snapshot were secured. Deletion confirmed.

## 7. Options for tomorrow — none started

| | option | local cost | GPU cost | what it buys |
|---|---|---|---|---|
| **A** | Retry unchanged on a different node | none | one pod | Fastest signal: works ⇒ node; fails identically ⇒ software |
| **B** ⭐ | **Add exact-production smoke + telemetry, then retry** | code + docs | one pod | Next fault costs **seconds, not a pod**, and is **diagnosable** |
| **C** | Simplify load path (explicit `.to("cuda")`, test raw base) | code | one pod | Isolates two open candidates — a guess without B's telemetry |
| **D** | Stop | none | none | Preserves integrity but **leaves Study 2 blocked**: §0 is required before further inference (`docs/study2-plan.md:45–49`) |

**Recommendation: B, then A on a different node**, preferring a datacenter GPU
over a community consumer card if the rate is acceptable.

**B is not a fix and must not be described as one.** The cause is unknown; B
makes failure cheap and legible. Doing A alone risks paying twice for the same
mute fault.

**Any of A/B/C requires a fresh written estimate and explicit owner approval
before spend.**

## 8. Money

**No charge is quotable for this attempt.** Billing is unsettled. The retained
`billing_pods.json` holds one RTX 4090 row (`2026-08-07`,
`0.23272516997531056`) whose values match `mining_pilot/cost_receipt.json`
exactly. **Attribution and settlement are both unestablished** — both runs fall
on Aug 7 local and straddle the UTC boundary, and provider date-grouping is
unknown, so a settled pull may *change* that row rather than add one.
**Do not quote it.** Billing deprioritised at the owner's direction.

## 9. Artifacts

`~/Documents/llama-tools-artifacts/probe-20260808/` — `study2/probe.log`,
`study2/study2_probe_multiple/run_manifest.json` (incomplete, 0 rows),
`probe_timing.txt`, `storage_mode.txt`, `env_fingerprint.json`,
`pip_freeze.txt`, `gpu.txt`, `image_tag.txt`, `bundle_sha256.txt`,
`reviewed_commit.txt`, `auto_terminate_attestation.txt`,
`pod_get_before_delete.json`, `billing_pods.json` (unsettled).

Storage was `ephemeral_pod_disk` with **no network volume** — everything above
survives only because it was copied off before deletion.
