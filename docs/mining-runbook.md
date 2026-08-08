# Mining stage runbook

Operational companion to `docs/prereg-study2.md`. **The preregistration governs;
this file only says how to execute what it already fixes.** Written after the
2026-08-07 pilot, from what actually went wrong.

## Before the pod exists

**1. Build and verify the bundle** (the repo is transferred as a git bundle, not
cloned — `origin` may lag the reviewed tip):

```bash
cd ~/Documents/llama-tools
git bundle create /tmp/llama-tools.bundle study2/on-policy-dpo
git bundle verify /tmp/llama-tools.bundle
shasum -a 256 /tmp/llama-tools.bundle | awk '{print $1}' > /tmp/llama-tools.bundle.sha256
```

## Creating the pod

**For the pilot** a Mac-side timer was adequate. **For calibration it is not** —
the run is hours, and `caffeinate -i` does not survive a closed lid. **Create the
pod with a provider-enforced deadline instead:**

```bash
runpodctl pod create --terminate-after '<UTC datetime>' ...
```

`--terminate-after` is enforced by RunPod and survives this machine sleeping,
losing network, or being closed. **`pod update` cannot add it to a
browser-created pod**, so it must be set at creation.

**The two flags take different units and are not the same value.**
`--terminate-after` is an **absolute UTC datetime**; the launcher's
`--provider-terminate-seconds` is an **integer duration in seconds**. Derive the
second from the first so the attestation is true:

```bash
# ILLUSTRATION — substitute the deadline the owner actually approved.
TERMINATE_AT='<APPROVED_UTC_DEADLINE>'          # e.g. YYYY-MM-DDTHH:MM:SSZ
runpodctl pod create --terminate-after "$TERMINATE_AT" ...
PROVIDER_SECONDS=$(( $(date -u -j -f '%Y-%m-%dT%H:%M:%SZ' "$TERMINATE_AT" +%s) - $(date -u +%s) ))
```

**This block is not runnable as written**, deliberately: a literal date here would
go stale and a copy-paste would arm a deadline in the past.

Pass `$PROVIDER_SECONDS` to the launcher. **Recompute it immediately before
launching**, not at pod creation — setup time has elapsed in between.

**Region must support network volumes.** `US-TX-1` does not; the pilot's first
deploy failed there. **Network volumes are region-locked** — a resume must launch
in the same region.

## After the pod exists — transfer the data the bundle cannot carry

**The bundle is code only. `data/processed/` and `eval/bfcl_data/` are git-ignored
and must be copied separately** — the pilot's first launch died on
`MinerError: mining pool is missing at data/processed/sft_dedup_v2.jsonl` because
this step did not exist. **Copy the pinned pool file, not the directory.** `data/processed/` is ~165 MB
of study-1 history; **the miner pins exactly one file from it**
(`sft_dedup_v2.jsonl`, §2.7). With `eval/bfcl_data/` at ~11 MB the transfer is
**~41 MB** rather than ~176 MB. **Create the destinations first:**

```bash
ssh -p "$PORT" -i "$KEY" root@"$IP" \
  'mkdir -p /root/llama-tools/data/processed /root/llama-tools/eval'
scp -P "$PORT" -i "$KEY" data/processed/sft_dedup_v2.jsonl \
  root@"$IP":/root/llama-tools/data/processed/
scp -P "$PORT" -i "$KEY" -r eval/bfcl_data/ root@"$IP":/root/llama-tools/eval/
```

**Both go to `/root` (container disk), never `/workspace`.** The network volume is
small and `df` reports the whole cluster, so it gives **no warning as it fills**.

**What is and is not verified — the scopes differ, so do not over-trust this.**

- **The miner** verifies the **pool** against `POOL_SHA256`, the **decontamination
  receipt**, the **amended manifest**, and re-derives the post-screen id digest,
  all before any model loads. A truncated pool transfer stops the run.
- **It does not hash every copied file.** The screened **question** files are read
  by the decontaminator; the **answer keys** are not verified at mining time.
- **The scorer's own gate** (`eval/dev_subset_gate.py`) verifies the
  `live_multiple` **questions and answer key** against their manifest digests —
  but that runs at **dev-baseline time**, not here. A corrupted answer key would
  surface then, not during mining.

## On the pod, before the miner

```bash
export HF_TOKEN=...                          # base model is gated, adapter private
export HF_HOME=/root/.cache/huggingface      # keep 16 GB of weights off the volume
python3 -m venv --system-site-packages .venv # preserve the image's CUDA torch
.venv/bin/pip install -r requirements-probe.txt
.venv/bin/python -m mining.mine_pairs --self-test
```

**Never `pip install` torch over the template's.** `requirements-probe.txt` omits
it deliberately.

## Running

**Always through `scripts/launch_mining_stage.py`** — the miner refuses a direct
model launch without the launcher's context token.

**`--fresh` on the first launch only.** It refuses if any evidence exists, so a
**resume is the identical command with `--fresh` removed**, and `HF_TOKEN`
re-exported if the session was lost.

**The miner logs one line per prompt.** A silent log means it is stuck, not busy.

## After

Copy the stage's **`--out-dir`** off **before** terminating — `mining_pilot/` for
the pilot, **`mining_out/` for calibration** (§2.4) — hash-verify it locally, then
terminate the pod, then delete the network volume separately — it bills until you
do and nothing prompts you. **Fetch the cost receipt** with
`runpodctl billing pods -o json`: an account-balance delta is not a billed amount.
