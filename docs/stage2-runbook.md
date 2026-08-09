# Stage 2 — one retry pod, RunPod UI + tmux

**Status: DRAFT, pending @codex review. Do not execute until both agents have signed off.**

Operator: the owner, driving the RunPod web console and a terminal.
Reviewed commit for this run: **`08be5d3b0c1293ac229fe7b04e2931f14bd149d0`** (on `main`).

Everything below is pasteable. `<ANGLE_BRACKETS>` are the only things you fill in, and
each one says where the value comes from. If a step's expected evidence does not appear,
**stop and post the output** — do not improvise a fix on a billing pod.

---

## 0. Agreed parameters

| | |
|---|---|
| Card | secure cloud, **A6000 or A100** — your choice, rate permitting |
| Approved envelope | **up to $4 per pod** (#general msg 2981), covering the initial pod and the step-1 second-node retry |
| Provider auto-terminate | **70 minutes** from pod creation, set **in the UI at creation** |
| Launcher deadline | provider deadline **minus 180 s** (= minute 67) |
| Planning basis | 1.1667 h × the live rate, plus any storage charge the console shows |
| Rate ceiling | $4 ÷ 1.1667 ≈ **$3.43/hr** compute-only; storage reduces this |

The wall-clock envelope behind those numbers: **16.8 min** bootstrap+download (measured
from the 2026-08-08 pod), **~10 min** ladder, **~31 min** generation on the pessimistic
6× assumption = 57.7 min, leaving ~9 min slack inside 67.

**No dollar figure is encoded in any script.** The provider deadline is the only bound
that survives this process being killed.

---

## 1. RunPod UI — create the pod

1. Secure Cloud → pick the card. **Write down the exact `$/hr` the console shows.** A
   remembered rate is not evidence about this pod.
2. Template: a PyTorch CUDA image. **Record the exact image tag** — the bootstrap fails
   if `RUNPOD_IMAGE_NAME` is unset or `unknown`, deliberately, because a run whose
   environment cannot be named cannot be reproduced.
3. Container disk **≥ 60 GB** (16 GB weights + venv + headroom). A volume is optional;
   if you attach one, use it as `--out-root` so evidence survives a stop.
4. **Set auto-terminate to 70 minutes at creation, before deploying.** This is the only
   bound that survives a SIGKILL. Note the **exact UTC termination timestamp** the console
   shows, and **write it down normalized to RFC 3339 UTC — `YYYY-MM-DDTHH:MM:SSZ`.** If the
   console displays local time, a relative countdown, or omits seconds, do the conversion
   here, on paper, where a mistake costs nothing. §6 rejects anything else and will not hand
   an unverified deadline to the launcher.
5. Deploy. **Billing starts now** — clone, venv, pip and the 16 GB download are all
   billed. There is no free setup phase.

**Record before continuing:** `<RATE>` (e.g. `0.79`), `<IMAGE_TAG>`,
`<TERMINATE_UTC>` (e.g. `2026-08-09T04:15:00Z`).

---

## 2. Connect and start tmux

Web terminal, or SSH from the console's connect details. Then:

```bash
tmux new-session -d -s probe
tmux attach -t probe
```

Detaching is `Ctrl-b d`; the run survives a dropped connection.

---

## 3. Fetch the reviewed commit (no file upload needed)

The repo is public as of PR #6, so nothing has to be transferred from your laptop.

```bash
cd /workspace
export RUNPOD_IMAGE_NAME='<IMAGE_TAG>'
export HF_HOME=/root/.cache/huggingface
mkdir -p "$HF_HOME"

git clone https://github.com/dipak-bhujbal/llama-tools.git src
git -C src bundle create /workspace/llama-tools.bundle --all
sha256sum /workspace/llama-tools.bundle | awk '{print $1}' > /workspace/llama-tools.bundle.sha256
```

**What the sidecar does and does not prove.** It detects mutation between bundle
creation here and the bootstrap reading it a minute later. It is **not** transfer
provenance — both sides are generated on this pod. The real integrity gate is: HTTPS
clone from GitHub → the bundle contains the reviewed SHA → the bootstrap detaches to
that exact SHA → `rev-parse` equality and a clean-tree assertion.

Record the source, so a later reader can retrace it:

```bash
mkdir -p /workspace/persist/study2
echo "https://github.com/dipak-bhujbal/llama-tools.git" > /workspace/persist/study2/clone_source_url.txt
git -C src rev-parse HEAD                                > /workspace/persist/study2/clone_initial_head.txt
```

---

## 4. HF token — typed, never an argument

```bash
read -rsp 'HF token: ' HF_TOKEN && echo && export HF_TOKEN
```

`read -rs` echoes nothing and takes no argument, so the value reaches neither the screen,
the shell history, nor any process's argv. **Do not** use `hf auth login --token $HF_TOKEN`;
the shell expands it before exec and the token lands in the new process's argv.

---

## 5. Bootstrap

```bash
cd /workspace
bash src/scripts/bootstrap_pod.sh \
  --bundle /workspace/llama-tools.bundle \
  --bundle-sha256-file /workspace/llama-tools.bundle.sha256 \
  --commit 08be5d3b0c1293ac229fe7b04e2931f14bd149d0 \
  --out-root /workspace/persist/study2 \
  --auto-terminate-set '<TERMINATE_UTC>@<RATE>'
```

**Expected evidence, in order:** `STEP 0` acknowledges the attestation · `STEP 3` prints
`bundle sha256 verified` · `STEP 4` prints `HEAD asserted: 08be5d3…` · `STEP 6` prints the
GPU name and CUDA version and `HF access OK` · `STEP 7` prints
`all 7 environment receipts asserted present and non-empty` · then `BOOTSTRAP COMPLETE`.

An `EVIDENCE GAP — locale provenance unavailable` line is **not** fatal by design; record
it and continue.

**Exit codes:** `64` usage · `65` the attestation is malformed — the provider deadline was
not set or not recorded · `66` bundle missing or hash mismatch · `67` clone landed on the
wrong SHA, or the bundle lacks the commit · `68` venv, versions, CUDA, HF or timeout
preflight failed. **Any non-zero: stop, post the output, terminate the pod.**

---

## 6. Derive the deadlines

**Enter the console's auto-terminate time normalized to RFC 3339 UTC — exactly
`YYYY-MM-DDTHH:MM:SSZ`.** If the console shows anything else (local time, "in 70 minutes",
no seconds), convert it yourself first. We do not guess the console's format, and the block
below **fails closed** rather than parsing something it does not recognise.

```bash
cd /workspace/llama-tools
TERMINATE_UTC='<TERMINATE_UTC>'    # e.g. 2026-08-09T04:15:00Z

unset PROVIDER_EPOCH DEADLINE_EPOCH
if [[ ! "$TERMINATE_UTC" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
  echo "ABORT: not RFC 3339 UTC — got '$TERMINATE_UTC', need YYYY-MM-DDTHH:MM:SSZ"
elif ! PROVIDER_EPOCH=$(date -u -d "$TERMINATE_UTC" +%s 2>/dev/null); then
  unset PROVIDER_EPOCH
  echo "ABORT: GNU date could not parse '$TERMINATE_UTC'"
else
  ROUND_TRIP=$(date -u -d "@$PROVIDER_EPOCH" +%Y-%m-%dT%H:%M:%SZ)
  if [[ "$ROUND_TRIP" != "$TERMINATE_UTC" ]]; then
    unset PROVIDER_EPOCH
    echo "ABORT: round-trip mismatch — entered '$TERMINATE_UTC', parsed back '$ROUND_TRIP'"
  else
    DEADLINE_EPOCH=$(( PROVIDER_EPOCH - 180 ))
    MINS_LEFT=$(( (DEADLINE_EPOCH - $(date -u +%s)) / 60 ))
    echo "round-trip OK: $ROUND_TRIP"
    echo "provider=$PROVIDER_EPOCH launcher=$DEADLINE_EPOCH now=$(date -u +%s)"
    echo "minutes of work left: $MINS_LEFT"
    [[ "$MINS_LEFT" -ge 40 ]] || echo "ABORT: only $MINS_LEFT min left, need >= 40"
  fi
fi
```

**Proceed only if you saw `round-trip OK` and no `ABORT` line.** `PROVIDER_EPOCH` is left
unset on every failure path, and §7 refuses to launch without it — so a misparsed timestamp
cannot silently become a deadline. The round-trip is what proves the epoch means the instant
the console displayed: compare the printed `round-trip OK` value against the console by eye.

**Why the ≥ 40 minute gate:** the launcher already refuses to start if its deadline is not
strictly earlier than the provider's, and refuses a deadline already past. Neither catches
the expensive case — enough time to *start* but not to *finish*. Launching into a run that
gets killed mid-generation is billed in full and yields nothing.

---

## 7. Launch — ladder gates the probe, same pod

**The launcher must run as a child of _this_ shell — not in a new tmux session.** This is
not a style preference; a second `tmux new-session` would break the run. See the box below.

```bash
cd /workspace/llama-tools

# Refuse to spend if the two prerequisites this shell carries are missing.
[[ -n "${HF_TOKEN:-}"        ]] || echo "ABORT: HF_TOKEN not set in this shell — redo §4"
[[ -n "${PROVIDER_EPOCH:-}"  ]] || echo "ABORT: §6 did not complete — no PROVIDER_EPOCH"
[[ -n "${DEADLINE_EPOCH:-}"  ]] || echo "ABORT: §6 did not complete — no DEADLINE_EPOCH"

# Never let §8 monitor a pid left behind by an earlier attempt.
rm -f /workspace/persist/study2/launcher.pid

nohup bash scripts/launch_probe.sh \
  --commit 08be5d3b0c1293ac229fe7b04e2931f14bd149d0 \
  --provider-deadline-epoch "$PROVIDER_EPOCH" \
  --deadline-epoch "$DEADLINE_EPOCH" \
  --out-root /workspace/persist/study2 \
  > /workspace/persist/study2/probe.log 2>&1 &
LAUNCHER_BG_PID=$!
echo "launched; shell-visible pid $LAUNCHER_BG_PID"
```

> **Why not `tmux new-session -d -s run …`.** §2 already started the tmux **server**. A
> session created later does not inherit this pane's exports — it inherits the *server's*
> environment, captured before §4 read the token. The base model is gated and the weights
> are **not** downloaded during bootstrap; the first `from_pretrained` happens inside the
> isolation ladder, i.e. inside the launcher. And §4 deliberately never writes the token to
> disk, so there is no cached credential to fall back on. A launcher started in a fresh tmux
> session would therefore reach ladder rung 1 with **no `HF_TOKEN` at all** and fail on a
> gated download — after billing had been running for minutes. Passing the token via
> `tmux new-session -e` or `tmux set-environment` would fix the environment and break §4
> instead, putting the secret into a command line visible in `ps`.

You are already inside tmux (session `probe`, §2), so `nohup … &` survives a dropped
connection exactly as a tmux session would. Detach with `Ctrl-b d`; reattach with
`tmux attach -t probe`.

The launcher runs, in this order: detached checkout + HEAD/clean assertions → acquire
fixtures → verify → **isolation ladder (the gate)** → verify again → generation
`multiple` → verify → generation `simple_python`. **The ladder is inside the launcher.**
If all four rungs pass, the probe proceeds automatically — no further approval, as agreed.

Watch it: `tail -f /workspace/persist/study2/probe.log`.

---

## 8. Liveness monitor — separate session, from the published PID

The launcher publishes its PID early, but not instantly. **Wait for the file rather than
racing it** — an empty read here yields a monitor watching nothing:

```bash
for _ in $(seq 1 60); do
  [[ -s /workspace/persist/study2/launcher.pid ]] && break
  sleep 1
done
LAUNCHER_PID=$(cat /workspace/persist/study2/launcher.pid 2>/dev/null || true)

[[ "$LAUNCHER_PID" =~ ^[0-9]+$ ]] \
  || echo "ABORT: no usable launcher.pid after 60s — check probe.log, do not start the monitor"
[[ "$LAUNCHER_PID" == "$LAUNCHER_BG_PID" ]] \
  || echo "ABORT: published pid $LAUNCHER_PID != launched pid $LAUNCHER_BG_PID — stale file or a second launcher"
```

Both checks must be silent before continuing. Then start the monitor in its own session —
it needs no credentials and no cwd, so the tmux environment trap from §7 does not apply
(absolute path used deliberately, since the tmux server's working directory is not this
pane's):

```bash
tmux new-session -d -s watch \
  "bash /workspace/llama-tools/scripts/probe_liveness.sh \
     --log /workspace/persist/study2/probe.log \
     --status-file /workspace/persist/study2/liveness.json \
     --pid $LAUNCHER_PID --interval 30"
```

**Monitor exit codes:** `0` alive · `70` finished, exit 0 · `71` finished, non-zero ·
**`72` = DIED HARD — process gone with no authenticated terminal record.** 72 means
SIGKILL, the OOM killer, or preemption. **On 72: stop the pod immediately and confirm in
the console that billing stopped.** A dead process cannot stop its own meter.

**Keep the monitor in its own `watch` session, never alongside the launcher.** The runbook
this replaces put both in one session, which is what made the old liveness check
self-satisfying: the monitor outlives the launcher and reports a session alive because the
monitor itself is keeping it so. The §7 structure — launcher as a plain background child of
the `probe` pane, monitor in a separate tmux session — makes that mistake impossible to
repeat by construction. The monitor's authority is `kill -0` on the PID; its
`--tmux-session` argument is context-only and is deliberately not passed here.

---

## 9. Ladder failure branches — decided in advance

The ladder aborts on the first failing rung and names the branch. Launcher exit **69** =
gate failed; the ladder's own JSON is at
`/workspace/persist/study2/isolation_ladder/isolation_ladder.json`, with `failed_at_step`.

**Read the launcher's exit code from its own terminal record, not from the shell:**

```bash
grep PROBE_EXIT_RECORD /workspace/persist/study2/probe.log
```

That line is written by the launcher's `EXIT` trap and carries `pid=` and `exit=`. **Check
the `pid=` matches `$LAUNCHER_PID`** — a log appended by two attempts would otherwise let
the first run's clean exit be read as the second's. `wait "$LAUNCHER_BG_PID"; echo $?` also
works, but only from the pane that launched it and only once. **No `PROBE_EXIT_RECORD` line
at all, with the process gone, is itself the finding:** the trap never ran, so the launcher
did not exit in an orderly way — that is the monitor's `72`, not an exit code to interpret.

| Fails at | Rung | Reading | Action |
|---|---|---|---|
| **1** | raw base, explicit `cuda:0`, no `device_map` | Not PEFT, not placement — node, driver or kernel | **Terminate. Retry once on a second node.** Reproduces on two nodes ⇒ software: pin a known-good Transformers/Torch pair for §0 only, at the **image boundary** (exact image tag + a bootstrap runtime-equality assertion), not via a torch overlay in `requirements-probe.txt` |
| **2** | raw base, `device_map="auto"` | `device_map="auto"` placement | Load explicitly on a single device |
| **3** | `PeftModel`, adapter **disabled** | The adapter-disabled base realization | Realize the base candidate as a **separately loaded raw base model**. **No prereg amendment is needed** — §0 does not specify the realization (confirmed independently by both agents), and `base_candidate_realization` is already written into every run manifest |
| **4** | `PeftModel`, adapter **enabled** | Contradicts the mining pilot, which ran this successfully | Node instability — change nodes |
| none | — | Environment was the fault | The probe runs automatically on this pod |

**Rung 3 vs 4 is an exploration, not a confirmation.** The pilot differs from the failed
probe in **torch** (2.8.0 vs 2.9.1) as well as adapter state, so it is not a clean control.

**Second-node retry (step-1 branch only):** terminate this pod, confirm billing stopped,
create a fresh pod on a different host, and repeat from §1. The $4 envelope is **per pod**,
so the retry is already approved — no new approval needed unless scope changes.

---

## 10. Artifacts — collect before terminating

```bash
ls -1 /workspace/persist/study2
sha256sum /workspace/persist/study2/study2_probe_*/generations.jsonl 2>/dev/null
python3 -c "import json;d=json.load(open('/workspace/persist/study2/study2_probe_multiple/run_manifest.json'));print(d['status'], d['rows_written'], '/', d['expected_rows'])"
```

Expect `complete` and `400 / 400` for `multiple`, `800 / 800` for `simple_python`.
A run that does not write exactly `n_prompts × n_candidates` rows is **incomplete** and its
numbers **may not be reported** (§0.5) — the script enforces this and exits non-zero.

**Reproduction check (§0.4): shipped SFT on `simple_python` is expected to score 369/400.**
Any other value is **stop-and-report**, not something to reconcile afterwards.

Pull everything down from your laptop:

```bash
scp -P <SSH_PORT> -r root@<POD_IP>:/workspace/persist/study2 ~/Documents/llama-tools-artifacts/probe-<DATE>/
```

---

## 11. Terminate and record the real cost

1. **Terminate the pod in the console. Confirm billing has stopped** — a killed process
   cannot stop its own meter, and the 2026-08-08 pod stayed allocated ~34 minutes after
   its run died.
2. Record, into the run evidence: actual elapsed, the actual hourly rate, the actual
   charge, and the billing-stopped confirmation.

**Do not quote a charge until a settled provider artifact exists.** The 08-08 attempt has
an unsettled row whose value is identical to the mining pilot's receipt and **cannot be
attributed** to either run; repeating that mistake is how the $0.2131 → $0.2327 error
happened.

---

## Abort criteria — any one of these ends the run

- Bootstrap exits non-zero (any code).
- Any `ABORT:` line printed by §6, §7 or §8 — every one of them is fail-closed by design.
- §6 does not print `round-trip OK`, or the round-tripped timestamp does not match the
  console by eye.
- Fewer than 40 minutes of work remain at §6.
- Launcher exits **69** (gate failed) → take the §9 branch.
- Monitor reports **72** (died hard) → stop the pod, confirm billing stopped.
- Reproduction check returns anything other than **369/400**.
- Anything you did not expect. Post the output; do not improvise on a billing pod.
