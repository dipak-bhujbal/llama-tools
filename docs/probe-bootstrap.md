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

`--terminate-after` takes an **absolute UTC datetime**; `launch_probe.sh` takes required
provider and script deadline epochs and bounds every paid command with `timeout`. **They are two
independent bounds** — the provider's survives this machine sleeping or the script being
SIGKILLed, the script's does not.

**Neither script knows what a dollar is, deliberately.** A ceiling is a *per-run approval*,
not a property of reusable source. Converting the approved ceiling and the pod's live rate
into seconds is the operator's job, done once, below, and recorded in the run evidence. When
the number lived in the source it went stale silently — and a stale cap in source is not an
*unenforced* budget, it is a **mechanically enforced one at a number nobody approved.**
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
2. Enable **auto-termination** at a deadline whose maximum charge is **≤ `$0.45`**.
   **Derive that hard ceiling from the rate the console actually shows for the pod you are
   creating** — do not carry a remembered rate. Round the affordable duration down to the
   provider's supported resolution, then subtract a small **provider-cap margin** for clock,
   rate-display and provider-rounding uncertainty. This margin protects the monetary ceiling.
   The launch-time **shutdown reserve** below is distinct: it leaves time to persist artifacts
   before provider SIGKILL. The two protections intentionally stack and must be recorded by
   name; do not silently treat either one as the other.
3. Save proof (a console screenshot) and a written record of the exact provider-cap margin
   chosen into the persistent volume.
**Do not derive the script deadline here.** It is derived at launch, below, from the
provider deadline you just set — see the next section for why the difference matters.

The script cannot verify this from inside the pod, by design — it has no account
credentials. It requires an explicit attestation string and records it as evidence.

## Pod: bootstrap

`--out-root` **must be a mounted persistent volume.** Container disk is destroyed on stop;
that is exactly how the study-1 pod lost `outputs/sft-full`. Evidence is written there from
the first preflight artifact onward, not copied at the end and hoped for.

```bash
export RUNPOD_IMAGE_NAME="<the exact template tag you launched>"

# Type the token at the prompt. Do not paste it into an `export HF_TOKEN=...`
# command and do not put it in a file.
#
# `export HF_TOKEN=<value>` writes the token into shell history in cleartext and
# echoes it on screen as you type or paste it, so it survives in scrollback, in
# any terminal recording, and in whatever you copied it from. A token written to
# a file on the mounted volume outlives the pod entirely. `read -rs` echoes
# nothing and the value never becomes a history entry, because it arrives as
# stdin rather than as part of a command line.
#
# (`export` is a shell builtin, so it does NOT appear in `ps` — an earlier
# version of this note claimed it did. The `ps` exposure is real for external
# commands that take the token as an argument, e.g. `hf auth login --token
# $HF_TOKEN`, where the shell expands the value into the new process's argv.)
# `&&`, not `;`: on EOF (Ctrl-D) or a closed stdin `read` fails, and an
# unconditional `export` would then publish an empty or stale value and let
# the run proceed on it. Export only on a successful read.
read -rsp 'HF read token: ' HF_TOKEN && echo && export HF_TOKEN

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
| 7 | `pip_freeze.txt`, `gpu.txt`, `image_tag.txt`, `env_fingerprint.json` (including locale variables/effective collation), `bundle_sha256.txt`, attestation → persistent root | — |

**An unknown image tag fails.** It is not recorded as `"unknown"` and treated as evidence —
a run whose environment cannot be named cannot be reproduced.

Locale provenance follows the telemetry-gap contract: if the image's `locale` resolver is
missing, fails, or returns malformed output, `env_fingerprint.json` records
`status: unavailable`, the reason, and `shell_resolved: null`; STEP 7 prints the gap and
continues. The liveness monitor is locale-independent, so this is an evidence gap rather
than a correctness failure, and it does not discard an otherwise valid billed bootstrap.

### About the dependency spec

`requirements-probe.txt` is a **new probe environment**, explicitly not study-1 provenance.
The pod that produced study 1 was terminated on 2026-07-21 with no dependency manifest
captured, so that environment is unrecoverable; any file claiming to be it would be
fabricated. The reviewer laptop is macOS/arm64 with CUDA unavailable, so freezing it would
be false as provenance and probably uninstallable on a CUDA pod.

**torch is deliberately not pinned** — it comes from the template's CUDA build, matched to
the pod's driver. Installing a different torch over it is the usual way to break a working
CUDA setup. The actual torch/CUDA/GPU versions are asserted and recorded at Step 6.

## Pod: §0 smoke gate — runs before the full probe

The 2026-08-08 §0 probe died at 64s with a CUDA illegal memory access and zero
generations, and the cause is still indeterminate. Relaunching the full probe would
spend the same money to learn the same nothing, because it varies four things at once.
Run the isolation ladder first. It varies one thing per rung:

**The ladder is itself billed.** It runs on a metered pod, loads an 8B model four times and
generates tokens. It reduces generation exposure — 4 loads and 32 generated tokens before
the 1,200-generation probe — but it is not free. Its time and cost still need a fresh
estimate and explicit approval before any pod starts, and it runs under the same absolute
wall-clock deadline as the full model-execution commands.

| Rung | Configuration | What an adjacent pair isolates |
|---|---|---|
| 1 | raw base, explicit `cuda:0`, no `device_map` | — |
| 2 | raw base, `device_map="auto"` | 1 vs 2 → **placement** |
| 3 | `PeftModel`, adapter **disabled** | 2 vs 3 → **the PEFT wrapper** |
| 4 | `PeftModel`, adapter **enabled** | 3 vs 4 → **adapter state** |

Rung 3 is what the probe's `base` candidate ran; rung 4 is what the mining pilot ran
successfully on 2026-08-07. One 610-token prompt, 8 new tokens per rung. It aborts at the
first failure and prints the verdict for that branch.

**`launch_probe.sh` runs the ladder itself**, as step 4, after fixture verification and
before the full probe, wall-clock bounded by the same shared deadline as the generations. A
non-green ladder exits `69` and the probe never starts. Fixtures are then verified *again*
between the gate and the first generation, preserving the invariant that a checksum check
sits immediately before each paid generation with nothing in between.
You do not run it manually as part of a probe, and there is no flag to skip it — a gate an
operator has to remember under time pressure on a billing pod is not a gate, and running it
inside the launcher is also what makes a green result same-run, same-node evidence rather
than a receipt from some earlier session on some other card.

```bash
# Inspect the plan — loads nothing, needs no GPU, reviewable off-pod.
.venv/bin/python eval/isolation_ladder.py --dry-run

# Standalone, only when diagnosing outside a probe launch. --out-dir is required:
# the rung that kills the process is the one whose evidence matters.
.venv/bin/python eval/isolation_ladder.py \
  --out-dir /workspace/persist/study2/isolation_ladder
```

The script re-execs itself with `CUDA_LAUNCH_BLOCKING=1`; do not set it yourself and do
not work around the re-exec. Setting that variable after the process has started is a
no-op that looks like it worked, and without it a CUDA fault surfaces at some later
synchronisation point and gets attributed to the wrong rung — which is exactly why the
postmortem could not name a cause.

It refuses to run unless the first prompt matches the pinned crash prompt by identity --
the SHA-256 of the rendered string and of its input ids, not a token count. That is
intended: a pass on some
other prompt is not evidence about the crash.

Evidence lands under `<out-root>/isolation_ladder/`:

| File | Written |
|---|---|
| `nvidia_smi_q_pre_run.txt` | once, before anything loads — the card's baseline |
| `telemetry/step0_pre_run.json` | once, with library versions |
| `telemetry/stepN_after_load.json` | after each load, **before** the dangerous call |
| `telemetry/stepN_after_generate.json` | after each successful generation — carries the rung's true peak VRAM |
| `telemetry/stepN_on_failure.json` | best effort on the rung that faults |
| `isolation_ladder.json` | written before rung 1 and rewritten at the start and end of **every** rung |

Every one of those is written atomically as it is produced, because the process being
measured is one that has already died abruptly once and taken all of its evidence with it.
A rung that kills the process still leaves the rungs before it, and its own after-load
snapshot, on disk.

The `on_failure` snapshot is the important one. torch's device queries will usually fail on
a poisoned CUDA context and come back `unavailable` — but `nvidia-smi` and the kernel log
are separate processes, so the **Xid code the driver just logged** is still readable, and
it is the single most diagnostic thing available.

Check `outcome` first. It is one of `all_passed`, `failed`, or **`incomplete`** — the last
meaning the ladder did not reach a conclusion, which is what a partial artifact left by a
hard death looks like. An incomplete artifact carries `INCOMPLETE — NO CONCLUSION` and names
the rung marked `running` as where it died. **Never read the absence of a failed rung as a
pass**; only `outcome: all_passed` is a pass, and only that exits 0.

`s0_reproduction` is separate from `failed_at_step` on purpose. The probe's realized
configuration is rung 3. A gated-repo 401 or disk-full error at rung 1 fails *before* that
configuration and is not the fault the probe recorded. The ladder therefore reports
`failed_at_or_before_probe_configuration`, `failed_at_probe_configuration`,
`failed_at_s0_phase`, and `fault_signature_matches_s0` as distinct facts. Exact reproduction
requires rung 3 plus the retained illegal-memory-access signature in the `generate` phase:
the original load and adapter attachment completed before the first prompt faulted. The same
fault on rung 1 or 2 is reported as `same_fault_before_probe_configuration`, which is useful
isolation evidence without being mislabelled as an exact reproduction. A rung-3 load-phase
illegal access is likewise a different failure despite sharing the error string.

A green ladder means only **not reproduced on this run**. The original failure happened on
this same first 610-token prompt, before later prompts or the second category ran, so a green
result does not point at prompt count, the later length range, or the second category. It
leaves intermittent or nondeterministic behavior and node/card/driver/environment differences
open; compare the telemetry rather than treating the original card or node as cleared.

Read `isolation_ladder.json` before deciding anything. **Fields that could not be measured
say so.** A consumer card has no ECC, and an unprivileged container usually cannot read the
kernel log for Xid codes; those come back `unavailable` with a reason, never as zero. Do
not read a missing measurement as a clean result.

### Watch the run with the liveness monitor, not with your eyes

Launch exactly as the **Pod: launch** section below specifies — that is the one authoritative
invocation, with the absolute deadlines and the shared `--out-root`. Do not retype it here.
Run it under tmux with its output teed to a log, then start the monitor from a second pane:

The launcher publishes its own PID to `<out-root>/launcher.pid` before any risky work.
Read it from there. Do **not** use `pgrep`: it cannot recover a launcher that has already
died — the case you most need the PID for — and it can match an unrelated process.

Use the literal persistent path in this pane. `PROBE_OUT_ROOT` is assigned inside a function
in the *launch* shell and is not exported, so referencing it in a second tmux pane silently
expands to empty and the monitor would watch `/probe.log`.

```bash
# Second pane. Literal path on the persistent volume — same value you passed as
# --out-root, written out in full because this shell never saw that variable.
OUT=/workspace/persist/study2

bash scripts/probe_liveness.sh \
  --log "${OUT}/probe.log" \
  --status-file "${OUT}/probe_status.json" \
  --pid "$(cat "${OUT}/launcher.pid")" \
  --tmux-session probe --interval 30
```

The PID file is the *source of the number*, never proof of life: a SIGKILLed process cannot
update a file, which is exactly how the 2026-08-08 probe came to have a PID file pointing at
nothing. The monitor still decides liveness with `kill -0` on that number.

`--pid` is required and is the **sole** liveness authority. `--tmux-session` is recorded as
context and never decides anything: you are told to run this monitor from a second pane of
that same session, so the session outlives the launcher by construction — treating "session
exists" as "run is alive" would report RUNNING forever after the death this exists to catch.

It asserts positively: the process is alive, or an **authenticated** terminal record exists
explaining why not, and that record carries an exit code. Authenticated means the
`PROBE_EXIT_RECORD` line in the log carries the PID being watched. A footer from a different
PID, or a prose-only footer with no PID at all, is *not* accepted — a log appended by two
consecutive runs would otherwise let the earlier run's clean exit be reported as this one's
outcome. Pass `--allow-legacy-footer` only for logs from a launcher predating that record.

Exit `72` means the process is gone with no authenticated record — SIGKILL, the OOM killer,
or host preemption. That is the one condition worth waking someone for, and the one the old
mtime-watching monitor could not detect. **Log staleness is reported but never alerts**: an
8B model load writes nothing for minutes while perfectly healthy.

Monitor exit codes: `0` alive · `70` exited 0 · `71` exited nonzero · `72` died hard.

## Pod: launch

### Derive the script deadline — immediately before launch, not earlier

**From the time remaining, never from the approved ceiling.** Bootstrap is billed: cloning,
the venv, the 16 GB download and every preflight all ran on the meter. Deriving from the
full ceiling and rate at this point would hand the script the *original* full duration a
second time, on top of what bootstrap already spent — the ceiling would be enforced twice
over and the provider would terminate mid-probe.

Run this on the pod, in the same shell, seconds before launching. The launcher receives
absolute epochs, not a relative duration, so pausing after derivation cannot silently move
either deadline later:

```bash
prepare_probe_deadlines() {
  # The absolute UTC deadline you set at pod creation.
  PROVIDER_TERMINATION="<THE_ISO8601_DEADLINE_YOU_SET>"
  SHUTDOWN_RESERVE_SECONDS=180
  PROBE_OUT_ROOT=/workspace/persist/study2
  TIMING_RECEIPT="${PROBE_OUT_ROOT}/probe_timing.txt"

  # A retry must never inherit usable-looking values from an earlier attempt.
  unset provider_termination_epoch SCRIPT_DEADLINE_EPOCH
  unset derivation_epoch remaining_seconds_at_derivation

  if [[ -e "${TIMING_RECEIPT}" ]]; then
    echo "Refusing to overwrite existing timing receipt: ${TIMING_RECEIPT}" >&2
    echo "Preserve it by renaming it with a UTC suffix; never delete it." >&2
    echo 'Example: mv -- "${TIMING_RECEIPT}" "${TIMING_RECEIPT}.$(date -u +%Y%m%dT%H%M%SZ).rejected"' >&2
    return 1
  fi
  if ! provider_termination_epoch=$(date -u -d "${PROVIDER_TERMINATION}" +%s); then
    echo "Invalid provider deadline; correct PROVIDER_TERMINATION and retry." >&2
    unset provider_termination_epoch
    return 1
  fi

  derivation_epoch=$(date -u +%s)
  SCRIPT_DEADLINE_EPOCH=$(( provider_termination_epoch - SHUTDOWN_RESERVE_SECONDS ))
  remaining_seconds_at_derivation=$(( SCRIPT_DEADLINE_EPOCH - derivation_epoch ))

  if (( remaining_seconds_at_derivation <= 0 )); then
    echo "No usable probe time remains before provider termination" >&2
    unset provider_termination_epoch SCRIPT_DEADLINE_EPOCH
    unset derivation_epoch remaining_seconds_at_derivation
    return 1
  fi

  if ! {
    echo "provider_termination=${PROVIDER_TERMINATION}"
    echo "provider_termination_epoch=${provider_termination_epoch}"
    echo "script_deadline_epoch=${SCRIPT_DEADLINE_EPOCH}"
    echo "derivation_epoch=${derivation_epoch}"
    echo "shutdown_reserve_seconds=${SHUTDOWN_RESERVE_SECONDS}"
    echo "remaining_seconds_at_derivation=${remaining_seconds_at_derivation}"
  } | tee "${TIMING_RECEIPT}"; then
    echo "Could not persist timing receipt; refusing to launch." >&2
    unset provider_termination_epoch SCRIPT_DEADLINE_EPOCH
    unset derivation_epoch remaining_seconds_at_derivation
    return 1
  fi
}

if ! prepare_probe_deadlines; then
  echo "Deadline derivation failed; correct the input and rerun this block. Do not launch." >&2
fi
```

This writes both deadlines, the derivation time, reserve and remaining time to the
persistent run evidence before launch. **The approved ceiling and the hourly rate stay
external** — they justified the provider deadline; they are not inputs to the script. The
snippet uses GNU `date`, intentionally: it runs on the Linux pod, not the macOS laptop.

**One shared deadline, not two halves.** `launch_probe.sh` gives each paid command whatever
is left before `SCRIPT_DEADLINE_EPOCH`. This matters
because the two commands are not the same size: `multiple` is 200 × 2 = **400 generations**,
`simple_python` is 400 × 2 = **800**. An even split would give the command with twice the
work the same allowance and reliably kill the probe mid-`simple_python`, after paying for it.
The provider epoch is also passed explicitly; the script refuses to start unless its shared
deadline is strictly earlier, so the two bounds are mechanically nested rather than merely
assumed to be.

```bash
cd llama-tools
tmux new-session -d -s probe \
  "bash scripts/launch_probe.sh \
     --commit <REVIEWED_40_CHAR_SHA> \
     --provider-deadline-epoch "${provider_termination_epoch}" \
     --deadline-epoch "${SCRIPT_DEADLINE_EPOCH}" \
     --out-root "${PROBE_OUT_ROOT}" 2>&1 | tee "${PROBE_OUT_ROOT}/probe.log""
```

**Detached (`-d`) and `tee`'d on purpose.** A foreground `tmux new -s probe` blocks the
pasted command sequence and captures no durable log; detaching survives a dropped SSH
connection and the log lands in the persistent root as evidence rather than scrollback.

## Pod: finish

1. Both category directories contain `generations.jsonl`, `report.md` and
   `run_manifest.json`, and each manifest reads `"status": "complete"` with its embedded
   on-disk ID×candidate validation passing.
2. `probe.log`, `probe_timing.txt` and the Step-7 environment files are alongside them in
   the persistent root.
3. **Stop the pod, then confirm in the console that billing stopped.** Record actual elapsed
   time and actual charge into the run evidence.

`launch_probe.sh` prints this inventory from an `EXIT` trap on **every** path — including
failure and timeout, which is when a still-billing pod is most likely to be abandoned.
