# Probe bootstrap — fresh pod, exact commit, capped spend

Executable path from a bare RunPod instance to a launch-ready checkout of an **exact
reviewed commit**, without publishing anything.

Governing constraint: the reviewed branch is **not on `origin`**, and putting it there is
publication — an owner action. This procedure therefore transfers the commit as a
**SHA-256-verified git bundle**, which needs no remote, no push, and no public repository.

Every step is `$0` until Step 5. **Step 4 is a required human action and the run must not
start without it.**

---

## Step 0 — build the bundle (owner's laptop, `$0`)

```bash
cd ~/Documents/llama-tools
git bundle create /tmp/study2.bundle study2/on-policy-dpo
git bundle verify /tmp/study2.bundle          # must say "records a complete history"
shasum -a 256 /tmp/study2.bundle              # record this value
```

A bundle is a single file containing real git objects. It clones like a remote, so the pod
gets the exact commit and its full history with nothing published. Typical size here is
~340 KB.

Transfer it to the pod with the `scp` command from the RunPod **Connect** tab, then verify
the hash **on the pod** before trusting it:

```bash
shasum -a 256 /workspace/study2.bundle        # must equal the value recorded above
```

## Step 1 — clone the exact commit (pod, `$0` beyond pod time)

```bash
cd /workspace
git clone -b study2/on-policy-dpo study2.bundle llama-tools
cd llama-tools
git rev-parse HEAD                            # must equal the reviewed SHA exactly
git status --porcelain                        # must print nothing
```

`launch_probe.sh` re-asserts both of these before it spends anything, so a mismatch here
fails closed rather than running the wrong code.

## Step 2 — environment (pod, `$0` beyond pod time)

`.venv/` is gitignored, so it does not travel in the bundle and must be created.

```bash
cd /workspace/llama-tools
python -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install \
  "torch==2.13.0" "transformers==5.14.1" "peft==0.19.1" \
  "accelerate==1.14.0" "huggingface-hub==1.24.0" "python-dotenv==1.2.2"
```

Exact direct versions, matching the environment the code was written and tested against. A
full transitive lockfile is deliberately **not** required for this probe — direct pins plus
the recorded `pip freeze` in Step 3 are enough to reproduce or diagnose it.

## Step 3 — capability preflight (pod, `$0`, fail closed)

Nothing below may be skipped. Each check exists because its absence breaks something after
money has been spent.

```bash
cd /workspace/llama-tools

# GNU timeout with --kill-after. The spend cap is enforced by it; without it
# launch_probe.sh refuses to start rather than run uncapped.
timeout --kill-after=1 1 true && echo "timeout OK"

# Exact version tuple actually importable — not just pip-resolvable.
.venv/bin/python - <<'PY'
from importlib.metadata import version
expected = {"torch": "2.13.0", "transformers": "5.14.1", "peft": "0.19.1",
            "accelerate": "1.14.0", "huggingface-hub": "1.24.0"}
bad = {p: (version(p), want) for p, want in expected.items() if version(p) != want}
assert not bad, f"version mismatch: {bad}"
import torch, transformers, peft, accelerate      # imports must actually work
assert torch.cuda.is_available(), "no CUDA device visible"
print("versions + imports OK:", torch.cuda.get_device_name(0), "| CUDA", torch.version.cuda)
PY

# HF access to the gated base model and the private SFT adapter.
.venv/bin/python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.repo_info("meta-llama/Llama-3.1-8B-Instruct",
              revision="0e9e39f249a16976918f6564b8830bc894c89659")
api.repo_info("centuriandip/llama-3.1-8b-tools-sft",
              revision="b6f4da479f8c6fc044ee8b802a92f47780f970c5")
print("HF access OK")
PY

# Record the environment into the run evidence.
mkdir -p eval/out
.venv/bin/pip freeze > eval/out/pip_freeze.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > eval/out/gpu.txt
echo "${RUNPOD_IMAGE_NAME:-unknown}" > eval/out/image_tag.txt
```

## Step 4 — provider spend cap ⚠️ REQUIRED HUMAN ACTION

**A process that has been SIGKILLed cannot stop its own billing.** The in-pod wall-clock
ceiling bounds *compute time*; it cannot bound *vendor charges* if the pod is forgotten or
the harness dies. Owner approval authorises spending at most `$2.50` — it does not
mechanically prevent a forgotten pod from exceeding it. Only a provider-side guard does.

Before Step 5, the owner must:

1. Record the **actual hourly rate** shown in the RunPod console and the **billing start
   time**.
2. Enable **auto-termination** (or an external watchdog) at a deadline whose maximum charge
   is **≤ `$2.50`**, keeping a reserve for persistence and shutdown.
   At `$0.44/hr` that is a deadline of **≤ 5 h 40 m**; subtract the reserve, so set
   **≤ 5 h 00 m**.
3. Save proof of the setting (a console screenshot is sufficient) into the run evidence.

Fill in before launching:

```
actual_gpu           : ____________________
actual_rate_usd_hr   : ____________________
billing_started_utc  : ____________________
auto_terminate_at    : ____________________   # <= $2.50 total, reserve included
proof_saved_to       : ____________________
```

## Step 5 — launch (the only paid step)

```bash
cd /workspace/llama-tools
tmux new -s probe
bash scripts/launch_probe.sh \
  --commit <REVIEWED_40_CHAR_SHA> \
  --usd-cap 2.50 \
  --usd-per-hour <ACTUAL_RATE_FROM_STEP_4> \
  --out-root /workspace/persist/study2 \
  --dry-run          # inspect the printed plan first, then re-run without --dry-run
```

`--out-root` **must be a mounted persistent volume**. Container disk is destroyed on stop —
that is how the study-1 pod lost `outputs/sft-full`. If no volume is mounted, the wrapper
packages and copies complete *and partial* outputs plus logs before termination, but a
mounted volume is the safer path.

## Step 6 — persist, validate, then stop

1. Confirm `generations.jsonl`, `report.md`, and `run_manifest.json` exist for **both**
   categories, and that each manifest reads `"status": "complete"` with its embedded
   on-disk ID×candidate validation passing.
2. Copy `pip_freeze.txt`, `gpu.txt`, `image_tag.txt`, and the tmux logs alongside them.
3. Verify hashes after the copy — a run is only "successful" once evidence is **validated
   on disk and durably persisted**. A console reminder is not persistence.
4. **Stop the pod, then confirm in the console that billing has stopped**, and record actual
   elapsed time and actual cost into the run evidence.

---

## Owner decisions required

| # | Decision | Why it cannot be an agent action |
|---|---|---|
| 1 | Commit transfer: bundle (above) or push the branch to `origin` | Publication is an owner action |
| 2 | Provider auto-termination at a `≤ $2.50` deadline | Only the account holder can set it, and no in-pod code can guarantee it |
| 3 | Explicit go for the paid run | Spend gate |
