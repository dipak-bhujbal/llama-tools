# Probe bootstrap — operator guide

> **Scope: the §0 endpoint qualification probe only.** Approved ceiling **`$0.45`**
> (#general msgs 2691–2697). **Calibration is NOT approved** and must not be run from this
> guide; it requires its own estimate, agent agreement and owner approval, after the probe's
> artifacts and the A1.2 stop-and-consult review.

Operator guide for `scripts/bootstrap_pod.sh`, which turns a bare RunPod instance into a
launch-ready checkout of an **exact reviewed commit**.

The checks live in the script, not in this document. Anything described here that is not
implemented and tested there is a bug in this document.

---

## ⚠️ Billing starts when the pod starts

There is **no unbilled setup phase.** Cloning, creating the venv, installing packages,
downloading 16 GB of weights and every preflight check are all billed pod time, drawn from
the **same `$0.45` lifecycle budget** as generation itself.

An earlier draft of this guide claimed "every step is `$0` until launch." That was wrong,
and it mattered: it implied the provider cap could be configured late.

**Consequence: provider auto-termination must already be set before the pod is running.**

**Set it at creation — it cannot be added afterwards.** `runpodctl pod update` does not
accept `--terminate-after`, so a browser-created pod cannot be retrofitted:

```bash
# ILLUSTRATION — substitute the deadline derived from the approved cap and the
# console's actual rate. A literal date here would go stale.
runpodctl pod create --terminate-after '<APPROVED_UTC_DEADLINE>' ...
```

`--terminate-after` takes an **absolute UTC datetime**; `launch_probe.sh` takes
`--usd-cap` and `--usd-per-hour` and derives its own wall-clock `timeout` from them. **They
are two independent bounds in different units** — the provider's survives this machine
sleeping, the script's does not.
It is Step 0 of the script, and the script refuses to proceed without an attestation of it.

## Transfer method: git bundle — decided, not open

Settled by the owner (msg 1980): **bundle now; push to `origin` only after Phase 0's
public-record repairs have been reviewed.**

The reasoning is worth keeping visible: `origin` is the repository the résumé links to.
Pushing today would publish the current state — which still contains the very errors Phase 0
exists to fix. A bundle is a point-to-point copy onto infrastructure the owner controls;
nothing becomes publicly readable, the no-publication rule stays intact, and the commit
still reaches the pod. The push to `origin` then happens once, carrying the repaired record.

**This guide does not carry a bundle hash.** A bundle is only meaningful at the tip that was
actually reviewed, and committing its hash would change `HEAD` and therefore change the
bundle — the circularity codex flagged. The hash travels as a **separate `.sha256` sidecar**
and enters the run evidence from outside the repository.

---

## Owner: build the bundle (laptop, after cycle-3 sign-off)

```bash
cd ~/Documents/llama-tools
REVIEWED_SHA=$(git rev-parse HEAD)          # the signed-off tip; record all 40 chars
git bundle create /tmp/llama-tools.bundle study2/on-policy-dpo
git bundle verify /tmp/llama-tools.bundle   # must report a complete history
shasum -a 256 /tmp/llama-tools.bundle | awk '{print $1}' \
  > /tmp/llama-tools.bundle.sha256
```

Send both files to the pod (`runpodctl send`, or `scp` from the Connect tab). They travel
together but are verified independently — the script checks the sidecar **before** cloning,
so no unverified object is ever trusted.

## Owner: set the provider cap ⚠️ before the pod runs

A process that has been SIGKILLed cannot stop its own billing, and approval to spend at
most `$0.45` authorises an amount rather than enforcing one. Only the provider-side deadline
enforces it.

1. Record the **actual hourly rate** from the RunPod console and the **billing start time**.
2. Enable **auto-termination** at a deadline whose maximum charge is **≤ `$0.45`**, keeping
   a reserve for persistence and shutdown. **Derive the ceiling from the rate the console
   actually shows for the pod you are creating** — do not carry a remembered rate. As a
   worked example only, at `$0.57/hr` the cap buys about 47 minutes, so set the
   deadline **strictly inside that — no more than 40 minutes** — leaving a reserve for
   artifact persistence and shutdown. **The general rule, not the example:**
   `deadline < $0.45 / actual_rate`, minus a shutdown reserve.
3. Save proof (a console screenshot) into the persistent volume.

The script cannot verify this from inside the pod, by design — it has no account
credentials. It requires an explicit attestation string and records it as evidence.

## Pod: bootstrap

`--out-root` **must be a mounted persistent volume.** Container disk is destroyed on stop;
that is exactly how the study-1 pod lost `outputs/sft-full`. Evidence is written there from
the first preflight artifact onward, not copied at the end and hoped for.

```bash
export RUNPOD_IMAGE_NAME="<the exact template tag you launched>"
export HF_TOKEN="<read token>"

bash scripts/bootstrap_pod.sh \
  --bundle /workspace/llama-tools.bundle \
  --bundle-sha256-file /workspace/llama-tools.bundle.sha256 \
  --commit <REVIEWED_40_CHAR_SHA> \
  --out-root /workspace/persist/study2 \
  --auto-terminate-set "<ISO8601-deadline-Z>@<rate>" \
  --dry-run                      # inspect, then re-run without --dry-run
```

What it enforces, in order — each check exists because of a way a run has failed or could
fail after the meter was already running:

| step | check | failure |
|---|---|---|
| 0 | provider auto-termination attested in `<ISO8601>@<rate>` form | exit 65 |
| 1 | `RUNPOD_IMAGE_NAME` set and not `unknown` | exit 68 |
| 2 | `--out-root` exists and is writable | exit 68 |
| 3 | bundle sha256 matches its sidecar, **before** clone | exit 66 |
| 4 | clone, detach at the SHA, assert `rev-parse HEAD` and a clean tree | exit 67 |
| 5 | venv created here; exact probe spec installed | — |
| 6 | GNU `timeout --kill-after` works; exact version tuple imports; CUDA visible; HF reaches the gated base **and** the private adapter | exit 68 |
| 7 | `pip_freeze.txt`, `gpu.txt`, `image_tag.txt`, `env_fingerprint.json`, `bundle_sha256.txt`, attestation → persistent root | — |

**An unknown image tag fails.** It is not recorded as `"unknown"` and treated as evidence —
a run whose environment cannot be named cannot be reproduced.

### About the dependency spec

`requirements-probe.txt` is a **new probe environment**, explicitly not study-1 provenance.
The pod that produced study 1 was terminated on 2026-07-21 with no dependency manifest
captured, so that environment is unrecoverable; any file claiming to be it would be
fabricated. The reviewer laptop is macOS/arm64 with CUDA unavailable, so freezing it would
be false as provenance and probably uninstallable on a CUDA pod.

**torch is deliberately not pinned** — it comes from the template's CUDA build, matched to
the pod's driver. Installing a different torch over it is the usual way to break a working
CUDA setup. The actual torch/CUDA/GPU versions are asserted and recorded at Step 6.

## Pod: launch

```bash
cd llama-tools
tmux new-session -d -s probe \
  "bash scripts/launch_probe.sh \
     --commit <REVIEWED_40_CHAR_SHA> \
     --usd-cap 0.45 \
     --usd-per-hour <ACTUAL_RATE> \
     --out-root /workspace/persist/study2 2>&1 | tee /workspace/persist/study2/probe.log"
```

**Detached (`-d`) and `tee`'d on purpose.** A foreground `tmux new -s probe` blocks the
pasted command sequence and captures no durable log; detaching survives a dropped SSH
connection and the log lands in the persistent root as evidence rather than scrollback.

## Pod: finish

1. Both category directories contain `generations.jsonl`, `report.md` and
   `run_manifest.json`, and each manifest reads `"status": "complete"` with its embedded
   on-disk ID×candidate validation passing.
2. `probe.log` and the Step-7 environment files are alongside them in the persistent root.
3. **Stop the pod, then confirm in the console that billing stopped.** Record actual elapsed
   time and actual charge into the run evidence.

`launch_probe.sh` prints this inventory from an `EXIT` trap on **every** path — including
failure and timeout, which is when a still-billing pod is most likely to be abandoned.
