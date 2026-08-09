# Stage 2 — one retry pod, RunPod UI + tmux

**Status: DRAFT, pending @claude review under the three-cycle role reversal. Do not execute
until both agents have signed off.**

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

### The wall-clock envelope, separated by what is actually known

An earlier version of this table said "**16.8 min** bootstrap+download". That was wrong in
its label and unsupported in its number: **bootstrap downloads no weights** — the first
`from_pretrained` is inside the launcher — and no surviving artifact from the 2026-08-08 pod
records bootstrap's duration, because bootstrap's own output was never captured to a file.
Corrected, and marked by provenance:

| Phase | Duration | Basis |
|---|---|---|
| Bootstrap — clone, venv, `pip install`, preflight | **not measured** | 2026-08-08 kept no bootstrap log. Dominated by the torch wheel install, so budget generously |
| Weight fetch, first load | **17 s fetch + 10 s load** | **measured**, `probe-20260808/study2/probe.log` — a warm/fast link on that node; do not assume it generalises |
| Isolation ladder, 4 rungs | **~10 min** | **assumption, upper bound.** Four model loads plus smoke generations; never measured end-to-end |
| Generation, 1,200 gens | **5.2 min** (1×) → **30.9 min** (6×) | **measured** anchor 0.2573 s/gen (mining pilot, 205.841 s / 800 gens); the 3×/6× multipliers are assumptions |

**Post-launch worst case ≈ 10 + 31 + 0.5 ≈ 41.5 min.** That is the number the launch floor
has to clear, and it is why the floor is **45 minutes, not 40**: at 40 the pessimistic path
does not fit at all.

**The 45-minute floor is a receipt, not a fresh choice.**
`probe-20260808/study2/probe_timing.txt` records `launch_floor_seconds=2700`, basis
`planning_40min_upper_plus_5min_buffer`. The 40 in the previous draft silently regressed a
recorded decision. **This runbook does not supersede it** — it restores it, and the
arithmetic above independently agrees.

**No dollar figure is encoded in any script.** The provider deadline is the only bound
that survives this process being killed.

**Known from the 2026-08-08 pod, for comparison, not for planning:** RTX 4090 at
**$0.351/hr**, image `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`, ephemeral pod disk.
That was a different card class from the A6000/A100 above and a *different reviewed commit*
(`2d8abdb`); it tells you the shape of a bill, not this run's rate.

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

**Record before continuing:** `<RATE>` (e.g. `0.79`), `<IMAGE_TAG>`,
`<TERMINATE_UTC>` (e.g. `2026-08-09T04:15:00Z`). **Do not click Deploy yet.**

### Capture the provider evidence **now, before you click Deploy**

Creation-time views do not survive the pod. A screenshot taken after termination cannot show
the rate you were quoted or the auto-terminate you set, so §11 cannot be where this is
collected. On your laptop, first:

```bash
mkdir -p ~/Documents/llama-tools-artifacts/probe-<DATE>/cost_evidence
```

Then capture into it, from the console, **before deploying**:

| File | What it must show |
|---|---|
| `01_rate_at_creation.png` (or `.jpg`) | the selected card and its **$/hr as displayed**, on the creation screen |
| `02_image_selected.png` (or `.jpg`) | the exact image tag being deployed |
| `03_auto_terminate_set.png` (or `.jpg`) | the auto-terminate setting **and** the resulting termination time |

These three are the only provider proof of what this run was priced at and bounded by. The
in-pod `auto_terminate_attestation.txt` receipt is **operator-entered** — it records what you
typed, and is not evidence of either. §11 collects the termination and settlement halves.

Verify the three files on your laptop before starting the meter. Edit `<DATE>` first:

```bash
bash <<'SH'
DEST=~/Documents/llama-tools-artifacts/probe-<DATE>/cost_evidence
if ! cd "$DEST"; then
  echo "ABORT: $DEST does not exist — do not deploy"
else
  BAD=()
  shopt -s nullglob
  for want in 01_rate_at_creation 02_image_selected 03_auto_terminate_set; do
    matches=( "$want"* )
    if [[ "${#matches[@]}" -ne 1 ]]; then
      BAD+=("must have exactly one capture: $want* (found ${#matches[@]})")
    elif [[ ! -f "${matches[0]}" || ! -s "${matches[0]}" ]]; then
      BAD+=("capture is non-regular or empty: ${matches[0]}")
    fi
  done
  if [[ "${#BAD[@]}" -eq 0 ]]; then
    echo "PRE-DEPLOY EVIDENCE READY (01-03 present and non-empty)"
  else
    printf 'ABORT: %s\n' "${BAD[@]}"
  fi
fi
SH
```

5. **Only after `PRE-DEPLOY EVIDENCE READY`: click Deploy.** Billing starts now — clone,
   venv, pip and the 16 GB download are all billed. There is no free setup phase. If the
   token did not print, do not deploy; fix the evidence capture while the creation view is
   still available.

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

**The clone is pinned to the reviewed SHA _before_ anything from it is executed.** A clone
of the default branch is a moving target: `main` advancing between review and run would
change the behaviour of the very script that enforces exact-SHA execution, while the runbook
still claimed the reviewed commit ran.

**Every command's exit status is captured, not just its output.** Checking values alone is
how `SOURCE PINNED` could print after `git bundle create` had failed: a valid HEAD and an
empty `git status` say nothing about whether the bundle was written.

```bash
cd /workspace
export RUNPOD_IMAGE_NAME='<IMAGE_TAG>'
export HF_HOME=/root/.cache/huggingface
mkdir -p "$HF_HOME" /workspace/persist/study2

REV=08be5d3b0c1293ac229fe7b04e2931f14bd149d0
P=/workspace/persist/study2
ERRS=()
try(){ local l="$1"; shift; "$@"; local s=$?; [[ $s -eq 0 ]] || ERRS+=("$l exit $s"); return $s; }
cap(){ local l="$1" v="$2"; shift 2; local o s; o=$("$@"); s=$?
       [[ $s -eq 0 ]] || ERRS+=("$l exit $s"); printf -v "$v" '%s' "$o"; return $s; }

if [[ -e /workspace/src ]]; then
  echo "ABORT: /workspace/src already exists."
  echo "       Inspect it and remove it yourself — this runbook does not delete"
  echo "       a directory it did not create."
else
  try clone  git clone https://github.com/dipak-bhujbal/llama-tools.git src \
    && try detach git -C src checkout --detach "$REV"

  cap rev-parse HEAD_NOW git -C src rev-parse HEAD
  [[ "$HEAD_NOW" == "$REV" ]] || ERRS+=("HEAD is '$HEAD_NOW', expected $REV")

  cap status TREE_DIRT git -C src status --porcelain
  [[ -z "$TREE_DIRT" ]] || ERRS+=("working tree not clean: $TREE_DIRT")

  try bundle-create git -C src bundle create /workspace/llama-tools.bundle --all
  try bundle-verify git -C src bundle verify /workspace/llama-tools.bundle
  [[ -s /workspace/llama-tools.bundle ]] || ERRS+=("bundle is missing or empty")

  cap bundle-sha BSHA bash -c \
    'sha256sum /workspace/llama-tools.bundle | awk "{print \$1}"'
  [[ "$BSHA" =~ ^[0-9a-f]{64}$ ]] || ERRS+=("bundle sha256 not a digest: '$BSHA'")
  try write-sha bash -c 'printf "%s\n" "$1" > /workspace/llama-tools.bundle.sha256' _ "$BSHA"

  cap remote-url ORIGIN_URL git -C src remote get-url origin
  [[ -n "$ORIGIN_URL" ]] || ERRS+=("origin URL is empty")

  try write-url  bash -c 'printf "%s\n" "$1" > "$2/clone_source_url.txt"'    _ "$ORIGIN_URL" "$P"
  try write-head bash -c 'printf "%s\n" "$1" > "$2/clone_detached_head.txt"' _ "$HEAD_NOW"   "$P"
  for f in "$P/clone_source_url.txt" "$P/clone_detached_head.txt" /workspace/llama-tools.bundle.sha256; do
    [[ -s "$f" ]] || ERRS+=("receipt not written or empty: $f")
  done

  if [[ "${#ERRS[@]}" -eq 0 ]]; then
    echo "SOURCE PINNED at $HEAD_NOW; tree clean; bundle verified; receipts written"
    echo "  origin: $ORIGIN_URL"
    echo "  bundle sha256: $BSHA"
  else
    printf 'ABORT: %s\n' "${ERRS[@]}"
  fi
fi
```

**Proceed only on `SOURCE PINNED`.** It now prints only after **every** step returned zero
*and* produced non-empty output — clone, detach, `rev-parse`, `status`, `bundle create`,
`bundle verify`, the hash, the origin URL, and all three receipt writes. `git bundle verify`
is included because a bundle can be written and still be unusable.

The bundle is created *after* the detach, so nothing is packaged from an unpinned tree;
`--all` still carries every ref, so it remains a complete mirror. The recorded URL is
`git remote get-url origin` — what the clone actually came from — not the string typed above,
which would agree with itself even if the clone had come from somewhere else.

**On a pre-existing `/workspace/src`, this refuses rather than deletes.** An earlier draft ran
an unconditional `rm -rf`, which on a reused or unexpected pod would destroy something the
operator had not looked at. Deleting is your call, not the runbook's.

**What the sidecar does and does not prove.** It detects mutation between bundle
creation here and the bootstrap reading it a minute later. It is **not** transfer
provenance — both sides are generated on this pod. The real integrity gate is: HTTPS
clone from GitHub → **detach and assert here, before any of it executes** → the bundle
contains the reviewed SHA → the bootstrap detaches to that same SHA → `rev-parse` equality
and a clean-tree assertion, re-checked in §5 immediately before execution.

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

The SHA is re-asserted here, immediately before the script runs, because §3 and §5 are
separate pastes and the gap between them is exactly where a tree can change:

```bash
cd /workspace
REV=08be5d3b0c1293ac229fe7b04e2931f14bd149d0
H=$(git -C /workspace/src rev-parse HEAD); h_st=$?
D=$(git -C /workspace/src status --porcelain); d_st=$?

if [[ "$h_st" -ne 0 ]]; then
  echo "ABORT: git rev-parse failed (exit $h_st) — redo §3, do not run bootstrap"
elif [[ "$d_st" -ne 0 ]]; then
  echo "ABORT: git status failed (exit $d_st) — redo §3, do not run bootstrap"
elif [[ "$H" != "$REV" ]]; then
  echo "ABORT: /workspace/src is at '$H', not the reviewed SHA — redo §3"
elif [[ -n "$D" ]]; then
  echo "ABORT: /workspace/src is dirty — redo §3, do not run bootstrap"
else
  bash src/scripts/bootstrap_pod.sh \
    --bundle /workspace/llama-tools.bundle \
    --bundle-sha256-file /workspace/llama-tools.bundle.sha256 \
    --commit "$REV" \
    --out-root /workspace/persist/study2 \
    --auto-terminate-set '<TERMINATE_UTC>@<RATE>'
fi
```

The `2>/dev/null` that used to sit on `git status` here is gone, and both statuses are now
tested. Suppressed stderr plus an empty result is indistinguishable from a clean tree: a
`git status` exiting 128 produced no output, which the old `-n` test read as "nothing dirty"
and fell straight through to bootstrap.

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
RATE='<RATE>'                      # the $/hr the console showed, e.g. 0.79
DDR=/workspace/persist/study2/deadline_derivation.txt

unset PROVIDER_EPOCH DEADLINE_EPOCH
if [[ ! "$RATE" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "ABORT: RATE must be the console's \$/hr as a bare number, e.g. 0.79 — got '$RATE'"
elif [[ ! "$TERMINATE_UTC" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
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
    NOW_EPOCH=$(date -u +%s)
    DEADLINE_EPOCH=$(( PROVIDER_EPOCH - 180 ))
    MINS_LEFT=$(( (DEADLINE_EPOCH - NOW_EPOCH) / 60 ))
    echo "round-trip OK: $ROUND_TRIP"
    echo "provider=$PROVIDER_EPOCH launcher=$DEADLINE_EPOCH now=$NOW_EPOCH"
    echo "minutes of work left: $MINS_LEFT"
    if [[ "$MINS_LEFT" -lt 45 ]]; then
      unset PROVIDER_EPOCH DEADLINE_EPOCH
      echo "ABORT: only $MINS_LEFT min of work left, need >= 45 — deadlines discarded"
    else
      NEW_DDR=$(printf '%s\n' \
        "schema=deadline_derivation/v1" \
        "provider_termination_utc=$TERMINATE_UTC" \
        "provider_termination_epoch=$PROVIDER_EPOCH" \
        "derivation_epoch=$NOW_EPOCH" \
        "launcher_deadline_epoch=$DEADLINE_EPOCH" \
        "shutdown_reserve_seconds=180" \
        "launch_floor_seconds=2700" \
        "launch_floor_basis=planning_40min_upper_plus_5min_buffer" \
        "launch_floor_source=probe-20260808/study2/probe_timing.txt" \
        "minutes_left_at_derivation=$MINS_LEFT" \
        "rate_per_hour=$RATE" \
        "rate_source=runpod_console_at_creation")
      if [[ -e "$DDR" ]] && ! printf '%s\n' "$NEW_DDR" | diff -q - "$DDR" >/dev/null; then
        unset PROVIDER_EPOCH DEADLINE_EPOCH
        echo "ABORT: $DDR exists and differs from this derivation. This will not"
        echo "       overwrite it. Read it; if it is genuinely superseded, keep it:"
        echo "         mv $DDR $DDR.superseded.\$(date -u +%s)"
        echo "       then re-run this block."
      elif printf '%s\n' "$NEW_DDR" > "$DDR.tmp.$$" && mv -f "$DDR.tmp.$$" "$DDR"; then
        echo "DEADLINES OK"
      else
        rm -f "$DDR.tmp.$$"
        unset PROVIDER_EPOCH DEADLINE_EPOCH
        echo "ABORT: could not write $DDR — refusing to launch without the receipt"
      fi
    fi
  fi
fi
```

**This step also writes the deadline-derivation receipt**, `deadline_derivation.txt`, which
§10 requires. It records the termination string and epoch, the derivation epoch, the launcher
deadline, the 180 s reserve, the 2700 s floor **with its basis and source**, and the rate with
where the rate came from. It is written atomically, and if a receipt from another run is
already present **this refuses rather than overwrites** — the alternative is a run whose
timing evidence silently belongs to a different pod.

An earlier draft required `probe_timing.txt` here instead. Nothing at `08be5d3` writes that
file — the launcher only *names* it in an exit listing — so on a fresh pod §10 could never
complete, and on a reused volume a stale one would have been misattributed to this run.

`RATE` is validated as a bare number before anything else, so an unsubstituted `<RATE>`
placeholder aborts here rather than being written into the receipt as if it were a price.

**Proceed only if you saw `round-trip OK` and `DEADLINES OK`, and no `ABORT` line.** Every
failure path — unfilled or non-numeric rate, bad shape, unparseable, round-trip mismatch, too
little time left, **and a receipt that cannot be written** — leaves `PROVIDER_EPOCH` and
`DEADLINE_EPOCH` unset, and §7
mechanically refuses to launch without both. The guard is not the `ABORT` message; the guard is that the values a launch
needs do not exist. The round-trip is what proves the epoch means the instant the console
displayed: compare the printed `round-trip OK` value against the console by eye.

**Why the ≥ 45 minute floor:** the launcher already refuses to start if its deadline is not
strictly earlier than the provider's, and refuses a deadline already past. Neither catches
the expensive case — enough time to *start* but not to *finish*. Launching into a run that
gets killed mid-generation is billed in full and yields nothing. 45 comes from
`probe-20260808/study2/probe_timing.txt` (`launch_floor_seconds=2700`) and matches §0's
~41.5 min pessimistic post-launch path. **§7 re-checks it immediately before launching**,
because §6 and §7 are separate pastes and the time between them is real.

---

## 7. Launch — ladder gates the probe, same pod

**The launcher must run as a child of _this_ shell — not in a new tmux session.** This is
not a style preference; a second `tmux new-session` would break the run. See the box below.

**The guards below enclose the launch.** They do not warn and fall through — if any
precondition is missing, the `nohup` line is never reached, because it lives inside the
`else`. Nothing is deleted and nothing is spent on a failed check.

```bash
unset LAUNCHER_BG_PID CANDIDATE_PID

if ! cd /workspace/llama-tools; then
  echo "ABORT: /workspace/llama-tools is unavailable. Not launching from another directory."
elif [[ ! -f scripts/launch_probe.sh || ! -s scripts/launch_probe.sh \
        || ! -r scripts/launch_probe.sh ]]; then
  echo "ABORT: scripts/launch_probe.sh is not a readable, non-empty regular file."
elif [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ABORT: HF_TOKEN not set in this shell — redo §4. Not launching."
elif [[ -z "${PROVIDER_EPOCH:-}" || -z "${DEADLINE_EPOCH:-}" ]]; then
  echo "ABORT: §6 did not complete — no verified deadlines. Not launching."
elif [[ "$DEADLINE_EPOCH" -ge "$PROVIDER_EPOCH" ]]; then
  echo "ABORT: launcher deadline is not strictly earlier than the provider's. Not launching."
elif [[ "$DEADLINE_EPOCH" -le "$(date -u +%s)" ]]; then
  echo "ABORT: launcher deadline has already passed. Not launching."
elif [[ $(( (DEADLINE_EPOCH - $(date -u +%s)) / 60 )) -lt 45 ]]; then
  echo "ABORT: only $(( (DEADLINE_EPOCH - $(date -u +%s)) / 60 )) min left at launch time,"
  echo "       floor is 45 (launch_floor_seconds=2700). Deadlines discarded; not launching."
  unset PROVIDER_EPOCH DEADLINE_EPOCH
else
  # Only now, past every guard: clear any pid from an earlier attempt and launch.
  if ! rm -f /workspace/persist/study2/launcher.pid; then
    echo "ABORT: could not clear the prior launcher.pid. Not launching."
  else
    nohup bash scripts/launch_probe.sh \
      --commit 08be5d3b0c1293ac229fe7b04e2931f14bd149d0 \
      --provider-deadline-epoch "$PROVIDER_EPOCH" \
      --deadline-epoch "$DEADLINE_EPOCH" \
      --out-root /workspace/persist/study2 \
      > /workspace/persist/study2/probe.log 2>&1 &
    CANDIDATE_PID=$!
    # A background launch returns a pid even when bash exits immediately. Give
    # fast failures time to settle, then make process liveness part of the token.
    sleep 1
    if kill -0 "$CANDIDATE_PID" 2>/dev/null; then
      LAUNCHER_BG_PID=$CANDIDATE_PID
      echo "LAUNCHED; shell-visible pid $LAUNCHER_BG_PID (alive after 1s settle)"
    else
      launcher_st=0
      wait "$CANDIDATE_PID" || launcher_st=$?
      unset LAUNCHER_BG_PID CANDIDATE_PID
      echo "ABORT: launcher exited during the 1s settle (exit $launcher_st)."
      if [[ -r /workspace/persist/study2/probe.log \
            && -s /workspace/persist/study2/probe.log ]]; then
        echo "       probe.log begins:"
        sed -n '1,20p' /workspace/persist/study2/probe.log \
          || echo "ABORT: probe.log became unreadable while collecting the diagnosis"
      else
        echo "ABORT: probe.log is absent, unreadable, or empty — no startup diagnosis"
      fi
      echo "       Not reporting LAUNCHED."
    fi
  fi
fi
```

**Proceed to §8 only on `LAUNCHED`.** `LAUNCHER_BG_PID` is unset first, so if the launch did
not happen — including a script that exists but exits immediately — there is no pid for §8
to compare against and §8 stops too. The path/readability check prevents the simplest false
start; the one-second liveness check is what prevents a present-but-fast-failing script from
printing the positive token. **`LAUNCHED` means only that the launcher is running after that
startup settle; it does not mean the run is healthy.** §8 is the continuing-health control.
The deadline guards duplicate checks the launcher already makes internally — deliberately,
so the failure costs a shell round-trip instead of a process start on a billing pod.

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

One block: wait for the pid, validate it, start the monitor, **and prove the monitor is
actually running**. The monitor start is enclosed by the same guards, so a bad pid cannot
leave a paid launcher running with a monitor that was never there.

```bash
unset LAUNCHER_PID
for _ in $(seq 1 60); do
  [[ -s /workspace/persist/study2/launcher.pid ]] && break
  sleep 1
done
LAUNCHER_PID=$(cat /workspace/persist/study2/launcher.pid 2>/dev/null || true)

if [[ -z "${LAUNCHER_BG_PID:-}" ]]; then
  echo "ABORT: §7 did not report LAUNCHED — nothing to monitor."
elif [[ ! "$LAUNCHER_PID" =~ ^[0-9]+$ ]]; then
  echo "ABORT: no usable launcher.pid after 60s — read probe.log. Monitor not started."
elif [[ "$LAUNCHER_PID" != "$LAUNCHER_BG_PID" ]]; then
  echo "ABORT: published pid $LAUNCHER_PID != launched pid $LAUNCHER_BG_PID"
  echo "       stale file or a second launcher. Monitor not started."
elif ! kill -0 "$LAUNCHER_PID" 2>/dev/null; then
  echo "ABORT: pid $LAUNCHER_PID is already gone — read probe.log. Monitor not started."
else
  tmux kill-session -t watch 2>/dev/null
  tmux new-session -d -s watch \
    "bash /workspace/llama-tools/scripts/probe_liveness.sh \
       --log /workspace/persist/study2/probe.log \
       --status-file /workspace/persist/study2/liveness.json \
       --pid $LAUNCHER_PID --interval 30"
  sleep 3
  if tmux has-session -t watch 2>/dev/null; then
    echo "MONITOR ACTIVE on pid $LAUNCHER_PID"
  else
    echo "ABORT: watch session did not survive startup — the run is UNMONITORED."
    echo "       Do not walk away. Fix the monitor or terminate the pod."
  fi
fi
```

**The run is monitored only if you saw `MONITOR ACTIVE`.** `tmux new-session -d` returns 0
for a command that exits immediately afterwards, so the session is checked a moment later
rather than trusted — an unmonitored paid run is precisely the failure this whole exercise
exists to prevent. The monitor needs no credentials and no cwd, so the §7 environment trap
does not apply to it; the absolute path is deliberate, since the tmux server's working
directory is not this pane's.

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
| **1** | raw base, explicit `cuda:0`, no `device_map` | Excludes PEFT, placement and adapter state — none is present. Implicates **the card, the driver, the torch/CUDA build, or the base weights at this revision** | **Terminate. Retry once on a second node** — that discriminates one host, it does not by itself convict or exonerate anything else. Reproducing on two nodes rules out **that one physical card and host**; it does **not** rule out the card *model*, the driver, or the image, all of which the two pods share. Next: pin a known-good Transformers/Torch pair for §0 only, at the **image boundary** (exact image tag + a bootstrap runtime-equality assertion), not via a torch overlay in `requirements-probe.txt` — and a different card model is a separate experiment, not a conclusion. The pinned base-weights revision stays open throughout |
| **2** | raw base, `device_map="auto"` | `device_map="auto"` placement | Load explicitly on a single device |
| **3** | `PeftModel`, adapter **disabled** | The adapter-disabled base realization | Realize the base candidate as a **separately loaded raw base model**. **No prereg amendment is needed** — §0 does not specify the realization (confirmed independently by both agents), and `base_candidate_realization` is already written into every run manifest |
| **4** | `PeftModel`, adapter **enabled** | Step 3 exonerates the wrapper; only the active LoRA weights differ. The mining pilot ran this same configuration successfully on 2026-08-07, so a failure is a **regression since that date — environment, card, or adapter revision** — not a standing defect in the code path | **Do not conclude "bad node."** Record which of the three it is by elimination: re-run on a different node *and* check the adapter revision actually loaded against the pilot's. A second node is the next diagnostic step, not the diagnosis |
| none | — | Environment was the fault | The probe runs automatically on this pod |

**Rung 3 vs 4 is an exploration, not a confirmation.** The pilot differs from the failed
probe in **torch** (2.8.0 vs 2.9.1) as well as adapter state, so it is not a clean control.

**Second-node retry (step-1 branch only):** terminate this pod, confirm billing stopped,
create a fresh pod on a different host, and repeat from §1. The $4 envelope is **per pod**,
so the retry is already approved — no new approval needed unless scope changes.

---

## 10. Artifacts — collect before terminating

**Run the acceptance check. It decides; you do not eyeball it.** It validates *both* runs and
computes the reproduction gate rather than asserting it in prose:

```bash
/workspace/llama-tools/.venv/bin/python - <<'PY'
import collections, hashlib, json, pathlib, subprocess, sys

ROOT = pathlib.Path("/workspace/persist/study2")
REPO = pathlib.Path("/workspace/llama-tools")
REV  = "08be5d3b0c1293ac229fe7b04e2931f14bd149d0"
PINS = REPO / "eval/manifests/bfcl_v4_study2.json"
RUNS = {"study2_probe_multiple": "multiple",
        "study2_probe_simple_python": "simple_python"}
CANDS = {"base", "sft"}
# What the launcher pins on the command line; asserted, not assumed.
EXPECT_MODEL = {
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    "base_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
    "sft_adapter": "centuriandip/llama-3.1-8b-tools-sft",
    "sft_adapter_subfolder": "adapter",
    "sft_adapter_revision": "b6f4da479f8c6fc044ee8b802a92f47780f970c5",
    "base_candidate_realization": "peft_model_with_adapter_disabled",
}
REPRO_DIR, REPRO_CAND, REPRO_WANT = "study2_probe_simple_python", "sft", 369

fail = []
rescored = {}   # dir -> {(id, candidate): overall_ok recomputed here}
def check(cond, msg):
    if not cond:
        fail.append(msg)


# The launcher asserted this before generation. Assert it again immediately
# before importing the parser/scorer: a post-run edit must not define the audit
# while the manifests continue to claim REV.
try:
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
except subprocess.CalledProcessError as e:
    detail = (e.stderr or e.stdout or "").strip()
    print(f"FAIL: cannot assert acceptance source (git exit {e.returncode}): {detail}")
    sys.exit(1)
if head != REV:
    print(f"FAIL: acceptance source HEAD {head!r} != reviewed {REV}")
    sys.exit(1)
if dirty:
    print(f"FAIL: acceptance source tree is dirty; refusing an unpinned re-score:\n{dirty}")
    sys.exit(1)

if not PINS.exists():
    print(f"FAIL: pin manifest absent at {PINS}"); sys.exit(1)
pins = json.loads(PINS.read_text())
spec_by_cat = {f["category"]: f for f in pins["files"] if f["role"] == "questions"}
key_by_cat  = {f["category"]: f for f in pins["files"] if f["role"] == "answer_key"}
# The release-commit key differs at simple_python_363 and scores 368, not 369.
# Naming it explicitly turns "wrong key" into a diagnosis instead of a mystery.
release_key = {f["category"]: f for f in pins["files"]
               if f["role"] == "answer_key_release_commit"}

# The pinned parser and the pinned scorer, so the recount cannot drift from the
# rule the run actually applied. bfcl_simple imports torch/transformers/peft;
# on this pod that is free of risk by construction — the launcher just ran it.
sys.path.insert(0, str(REPO / "eval"))
try:
    from bfcl_scoring import score
    from bfcl_simple import extract_json
except Exception as e:                                    # noqa: BLE001
    print(f"FAIL: cannot import the pinned scorer/parser from {REPO/'eval'}: {e}")
    sys.exit(1)


def verify_pinned(spec, label, fail):
    """Return (path, ids) after checking bytes and id digest against the pin."""
    p = REPO / spec["local_path"]
    if not p.exists():
        fail.append(f"{label}: pinned file absent at {p}")
        return None, None
    raw = p.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != spec["sha256"]:
        fail.append(f"{label}: sha256 {got} != pinned {spec['sha256']}")
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    ids = [str(r["id"]) for r in rows]
    dig = hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
    if dig != spec["sorted_id_sha256"]:
        fail.append(f"{label}: sorted-id digest {dig} != pinned {spec['sorted_id_sha256']}")
    if len(ids) != spec["row_count"]:
        fail.append(f"{label}: {len(ids)} rows != pinned {spec['row_count']}")
    return rows, ids

for d, cat in RUNS.items():
    run, spec = ROOT / d, spec_by_cat[cat]
    man_p, gen_p = run / "run_manifest.json", run / "generations.jsonl"
    if not man_p.exists() or not gen_p.exists():
        fail.append(f"{d}: MISSING " + ", ".join(
            p.name for p in (man_p, gen_p) if not p.exists()))
        continue

    # --- expected IDs and the scoring key, both verified against the pins ----
    _, ids = verify_pinned(spec, f"{cat}/questions", fail)
    if ids is None:
        continue
    kspec = key_by_cat[cat]
    key_rows, _ = verify_pinned(kspec, f"{cat}/answer_key", fail)
    if key_rows is None:
        continue
    gt_by_id = {str(r["id"]): r["ground_truth"][0] for r in key_rows}
    expected_pairs = {(i, c) for i in ids for c in CANDS}

    # --- provenance the manifest must agree with -----------------------------
    m = json.loads(man_p.read_text())
    check(m.get("status") == "complete",
          f"{d}: status={m.get('status')!r}, expected 'complete'")
    check(m.get("code_revision") == REV,
          f"{d}: code_revision={m.get('code_revision')!r}, expected {REV}")
    check(m.get("category") == cat,
          f"{d}: category={m.get('category')!r}, expected {cat!r}")
    check(m.get("candidates") == ["base", "sft"],
          f"{d}: candidates={m.get('candidates')!r}, expected ['base', 'sft']")
    check(m.get("decoding") == {"do_sample": False, "max_new_tokens": 512},
          f"{d}: decoding={m.get('decoding')!r}, expected greedy/512")
    for k, v in EXPECT_MODEL.items():
        check(m.get(k) == v, f"{d}: {k}={m.get(k)!r}, pinned {v!r}")
    inputs = m.get("inputs") or {}
    check((inputs.get("questions") or {}).get("sha256") == spec["sha256"],
          f"{d}: manifest questions sha256 != pinned {spec['sha256']}")
    man_key_sha = (inputs.get("answer_key") or {}).get("sha256")
    check(man_key_sha == kspec["sha256"],
          f"{d}: manifest answer_key sha256 {man_key_sha} != canonical {kspec['sha256']}")
    if cat in release_key and man_key_sha == release_key[cat]["sha256"]:
        fail.append(f"{d}: scored against the RELEASE-COMMIT answer key, not the "
                    f"canonical one — this is the 369-vs-368 key difference")
    check(m.get("expected_rows") == len(expected_pairs),
          f"{d}: expected_rows={m.get('expected_rows')}, pins imply {len(expected_pairs)}")
    check(m.get("rows_written") == m.get("expected_rows"),
          f"{d}: rows_written={m.get('rows_written')} != expected_rows={m.get('expected_rows')}")
    check((m.get("validation") or {}).get("ok") is True,
          f"{d}: manifest validation.ok is not true")

    # --- coverage recounted from disk; the manifest is corroboration only ----
    rows = [json.loads(l) for l in gen_p.read_text().splitlines() if l.strip()]
    pairs = collections.Counter((str(r["id"]), r["model_name"]) for r in rows)
    dupes = [p for p, c in pairs.items() if c > 1]
    check(not dupes, f"{d}: {len(dupes)} duplicated (id,candidate) pairs, e.g. {dupes[:3]}")
    missing = expected_pairs - set(pairs)
    extra   = set(pairs) - expected_pairs
    check(not missing,
          f"{d}: {len(missing)} EXPECTED pairs absent, e.g. {sorted(missing)[:3]}")
    check(not extra,
          f"{d}: {len(extra)} UNEXPECTED pairs present, e.g. {sorted(extra)[:3]}")
    check(len(rows) == len(expected_pairs),
          f"{d}: {len(rows)} rows on disk, expected {len(expected_pairs)}")

    # --- re-score from the raw output with the pinned parser + canonical key --
    # Not a re-read of `overall_ok`: the stored flag is whatever key and rule
    # the run applied. This recomputes the verdict and then requires the two to
    # agree, so a run scored against a different key fails here rather than
    # being reported.
    here, disagree = {}, []
    for r in rows:
        rid = str(r["id"])
        gt = gt_by_id.get(rid)
        if gt is None:
            continue                      # already reported as an unexpected pair
        _, _, ok, why = score(extract_json(r.get("output") or ""), gt)
        here[(rid, r["model_name"])] = ok
        stored = r.get("overall_ok")
        if type(stored) is not bool:
            disagree.append(f"{rid}/{r['model_name']} stored overall_ok is not a JSON "
                            f"boolean: {stored!r}")
        elif ok != stored:
            disagree.append(f"{rid}/{r['model_name']} stored={stored!r} "
                            f"recomputed={ok} ({why or 'ok'})")
    rescored[d] = here
    check(not disagree,
          f"{d}: {len(disagree)} rows re-score differently against the canonical "
          f"key — the stored verdicts were not produced by this rule/key, e.g. "
          f"{disagree[:3]}")

# --- reproduction gate, recomputed -------------------------------------------
spec = spec_by_cat[RUNS[REPRO_DIR]]
if REPRO_DIR not in rescored:
    fail.append(f"reproduction gate: {REPRO_DIR} did not reach re-scoring")
else:
    sub = [ok for (_, c), ok in rescored[REPRO_DIR].items() if c == REPRO_CAND]
    got = sum(1 for ok in sub if ok)
    print(f"reproduction: {REPRO_CAND} on {REPRO_DIR} re-scored {got}/{len(sub)} "
          f"(prereg §0.4 expects {REPRO_WANT}/{spec['row_count']})")
    check(len(sub) == spec["row_count"],
          f"reproduction gate: {len(sub)} {REPRO_CAND} rows, expected {spec['row_count']}")
    check(got == REPRO_WANT,
          f"REPRODUCTION GATE FAILED: re-scored {got}, prereg §0.4 expects {REPRO_WANT}")

if fail:
    print("\n".join("FAIL: " + f for f in fail))
    print(f"\n{len(fail)} FAILURE(S) — these numbers may not be reported (§0.5).")
    sys.exit(1)
print("ACCEPTANCE PASS: both runs complete, IDs match the pinned set exactly, "
      "provenance pinned, reproduction gate met.")
PY
echo "acceptance exit: $?"
```

**`acceptance exit: 0` and `ACCEPTANCE PASS` are the only results that let these numbers be
reported.**

The acceptance step uses the repository's exact `.venv` interpreter, not ambient `python3`.
Importing `bfcl_simple` pulls in torch/Transformers/PEFT; the launcher proved those packages
exist in `.venv`, not in the image's unrelated system interpreter.

**It checks identity, not just cardinality.** An earlier version compared counts — rows,
unique pairs, candidate names — which 200 *entirely wrong* IDs would satisfy perfectly. The
expected ID set is now derived from the pinned questions file, after that file's own
`sha256` and `sorted_id_sha256` are verified against `eval/manifests/bfcl_v4_study2.json`,
and the run must cover **exactly** that set × `{base, sft}` — missing and unexpected pairs
are reported separately. Duplicates are still checked apart from the total, because a
duplicate can mask a missing pair while the count still looks right.

**Provenance is asserted live, not inherited.** Immediately before importing the parser and
scorer, the acceptance step re-asserts that `/workspace/llama-tools` is at the reviewed SHA
with a clean tree. `code_revision`, the base model and revision, and the adapter
repo/subfolder/revision must match what the launcher pins; the manifest's own `validation`
block is treated as corroboration, since it records what a past process concluded, and disk
is recounted here regardless.

**The score itself is recomputed, not read back.** Pinning the questions proves *which items*
were asked; it says nothing about the key that decided *what counted as right*, and here that
distinction has a name — the retained release-commit key differs at `simple_python_363` and
scores **368, not 369**. So the canonical answer key is verified byte-for-byte and by sorted-id
digest against the pin manifest, the run manifest's `inputs.answer_key.sha256` must equal that
same canonical hash, scoring against the release-commit key is called out by name rather than
just failing, and then every row is **re-parsed from its raw `output` with the pinned
`extract_json` and re-scored with the pinned `score()`**. The stored `overall_ok` must be an
actual JSON boolean and agree exactly with that recount row by row — missing values, `0`/`1`,
and truthy strings are rejected rather than coerced. The reproduction figure is the
**recomputed** one, with its
denominator taken from the pin rather than hard-coded — a run scored against the wrong key
cannot reach `ACCEPTANCE PASS` by carrying its own verdicts.

Then hash the required artifacts — **by explicit list, failing loudly on absence.** A glob
with `2>/dev/null` cannot distinguish "hashed everything" from "matched nothing":

**First, wait for the monitor to finish.** `liveness.json` is rewritten — atomically, via
`mv` — when the monitor notices the terminal record. §10 can legitimately run the moment
`PROBE_EXIT_RECORD` appears, while the monitor is still mid-sleep reporting `running`;
hashing it then produces a manifest that is false seconds later, before the `scp`.

```bash
unset MONITOR_TERMINAL_OK MONITOR_TERMINAL_PID
for _ in $(seq 1 60); do
  tmux has-session -t watch 2>/dev/null || break
  sleep 5
done
if tmux has-session -t watch 2>/dev/null; then
  echo "ABORT: the watch session is still alive after 5 min — do not hash a live"
  echo "       liveness.json. The monitor exits on its first non-alive verdict, so"
  echo "       either the launcher is still running (read probe.log and wait) or the"
  echo "       session is being held open by something else (tmux ls; tmux capture-pane"
  echo "       -pt watch). Do not proceed until it is gone."
else
  CURRENT_LAUNCHER_PID=$(cat /workspace/persist/study2/launcher.pid 2>/dev/null || true)
  if [[ ! "$CURRENT_LAUNCHER_PID" =~ ^[0-9]+$ ]]; then
    echo "ABORT: launcher.pid is missing or non-numeric — no current run identity"
  elif [[ -n "${LAUNCHER_PID:-}" && "$LAUNCHER_PID" != "$CURRENT_LAUNCHER_PID" ]]; then
    echo "ABORT: shell launcher pid $LAUNCHER_PID != file pid $CURRENT_LAUNCHER_PID"
  elif python3 - "$CURRENT_LAUNCHER_PID" \
           /workspace/persist/study2/liveness.json \
           /workspace/persist/study2/deadline_derivation.txt <<'PY'
import datetime, json, sys
pid, path, deadline_path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(path))
except Exception as e:
    print(f"ABORT: cannot read liveness.json: {e}"); sys.exit(1)
try:
    receipt = {}
    for line in open(deadline_path):
        key, value = line.rstrip("\n").split("=", 1)
        receipt[key] = value
    derivation_epoch = int(receipt["derivation_epoch"])
    provider_epoch = int(receipt["provider_termination_epoch"])
    checked_epoch = int(datetime.datetime.strptime(
        d["checked_at_utc"], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=datetime.timezone.utc).timestamp())
except Exception as e:
    print(f"ABORT: cannot tie liveness to deadline_derivation.txt: {e}"); sys.exit(1)

checks = {
    "liveness schema": d.get("schema") == "probe_liveness/v1",
    "state": d.get("state") == "exited",
    "exit_code": d.get("exit_code") == 0,
    "watched_pid": str(d.get("watched_pid")) == str(pid),
    "authenticated footer": d.get("footer_state") == "complete",
    "pid no longer alive": d.get("pid_alive") is False,
    "no terminal alert": d.get("alert") is False,
    "deadline schema": receipt.get("schema") == "deadline_derivation/v1",
    "same-run lower clock bound": checked_epoch >= derivation_epoch,
    "provider upper clock bound": checked_epoch <= provider_epoch,
}
bad = [name for name, ok in checks.items() if not ok]
if bad:
    print("ABORT: liveness is not terminal-clean/current: " + ", ".join(bad))
    sys.exit(1)
print(f"MONITOR TERMINAL OK (pid={pid}; checked_epoch={checked_epoch}; "
      f"derivation_epoch={derivation_epoch})")
PY
  then
    MONITOR_TERMINAL_OK=1
    MONITOR_TERMINAL_PID=$CURRENT_LAUNCHER_PID
  else
    unset MONITOR_TERMINAL_OK MONITOR_TERMINAL_PID
  fi
fi
```

**`MONITOR TERMINAL OK` is required before hashing.** It also proves the terminal record
belongs to *this* PID and was written no earlier than this run's deadline derivation, so a
matching stale `launcher.pid` + `liveness.json` pair from an earlier pod cannot be accepted.
The shell variables set on success mechanically gate the next block; opening a new shell
requires re-running this validation rather than carrying the token by memory.

```bash
CURRENT_LAUNCHER_PID=$(cat /workspace/persist/study2/launcher.pid 2>/dev/null || true)
if [[ "${MONITOR_TERMINAL_OK:-}" != 1 \
      || ! "${MONITOR_TERMINAL_PID:-}" =~ ^[0-9]+$ \
      || "$MONITOR_TERMINAL_PID" != "$CURRENT_LAUNCHER_PID" ]]; then
  echo "ABORT: terminal monitor validation is absent or no longer matches launcher.pid"
elif ! cd /workspace/persist/study2; then
  echo "ABORT: evidence root missing — not hashing whatever directory this shell is in"
else
REQUIRED=(
  study2_probe_multiple/generations.jsonl       study2_probe_multiple/run_manifest.json
  study2_probe_multiple/report.md
  study2_probe_simple_python/generations.jsonl  study2_probe_simple_python/run_manifest.json
  study2_probe_simple_python/report.md
  isolation_ladder/isolation_ladder.json        isolation_ladder/nvidia_smi_q_pre_run.txt
  pip_freeze.txt  gpu.txt  image_tag.txt  auto_terminate_attestation.txt
  reviewed_commit.txt  env_fingerprint.json  bundle_sha256.txt
  deadline_derivation.txt  launcher.pid  liveness.json  probe.log
  clone_source_url.txt  clone_detached_head.txt
)
BAD=(); TMP=$(mktemp ./artifact_sha256.XXXXXX) || BAD+=("could not create temp file")

for f in "${REQUIRED[@]}"; do
  if [[ ! -s "$f" ]]; then BAD+=("missing or empty: $f"); continue; fi
  sha256sum "$f" >> "$TMP" || BAD+=("sha256sum failed (exit $?): $f")
done

# Ladder telemetry is a variable file set: hash whatever is there, require >=1.
# find's own status must be captured BEFORE sorting -- piping into sort would
# report sort's status, so a partial traversal would look like a complete one.
TELE_RAW=$(find isolation_ladder/telemetry -type f); find_st=$?
[[ "$find_st" -eq 0 ]] || BAD+=("find on isolation_ladder/telemetry failed (exit $find_st)")
TELE=$(printf '%s\n' "$TELE_RAW" | sort)
N_TELE=$(printf '%s' "$TELE_RAW" | grep -c . )
[[ "$N_TELE" -ge 1 ]] || BAD+=("no files under isolation_ladder/telemetry/")
while IFS= read -r t; do
  [[ -n "$t" ]] && { sha256sum "$t" >> "$TMP" || BAD+=("sha256sum failed: $t"); }
done <<< "$TELE"

WANT=$(( ${#REQUIRED[@]} + N_TELE ))
GOT=$(grep -c . "$TMP")
[[ "$GOT" -eq "$WANT" ]] || BAD+=("hashed $GOT entries, expected $WANT")

if [[ "${#BAD[@]}" -eq 0 ]]; then
  if mv -f "$TMP" artifact_sha256.txt; then
    echo "ARTIFACTS COMPLETE ($GOT entries)"
  else
    rm -f "$TMP"
    echo "ABORT: could not install artifact_sha256.txt"
  fi
else
  rm -f "$TMP"; printf 'ABORT: %s\n' "${BAD[@]}"
fi
fi
```

**Why this shape.** The previous version decided by grepping its own output for a marker
string, so a `sha256sum` that *failed* wrote nothing, matched no marker, and printed
`ARTIFACTS COMPLETE`; a failed read of the file had the same shape, because `grep`'s exit 2
also lands in the success branch. Now every hash command's status is checked, the entry count
must equal the expected count exactly, and the manifest is written to a temp file and
`mv`-ed into place only on full success — so a partial run leaves the previous manifest
intact and is safe to re-run. `artifact_sha256.txt` is not in its own list.

The `cd` is now genuinely enclosing — the whole block sits in its `else`. Previously it was
`cd … || { echo ABORT; }`, which prints and then carries on, creating and hashing files in
whatever directory the shell happened to be in.

**The list is the evidence inventory, not a sample.** It carries both `report.md` files, all
seven environment receipts the bootstrap asserts (including `reviewed_commit.txt`), §6's
`deadline_derivation.txt`, the launcher's `launcher.pid`, the monitor's `liveness.json`, the
ladder JSON with its pre-run SMI capture and telemetry, and the §3 clone receipts.

**Two files are deliberately absent, for the same reason.** `storage_mode.txt` and
`probe_timing.txt` both appear in the 2026-08-08 artifacts, but **no script writes either at
`08be5d3`** — the launcher only names `probe_timing.txt` in an exit listing. Requiring a file
nothing creates makes `ARTIFACTS COMPLETE` unreachable on a fresh pod, and satisfiable by a
stale file on a reused volume. §6's `deadline_derivation.txt` replaces it and is written by
this runbook, for this run.

Pull everything down from your laptop:

```bash
scp -P <SSH_PORT> -r root@<POD_IP>:/workspace/persist/study2 ~/Documents/llama-tools-artifacts/probe-<DATE>/
```

---

## 11. Terminate and record the real cost

**Pull the artifacts down before terminating** (§10's `scp`), then terminate. Everything
below lands on **your laptop**, in `~/Documents/llama-tools-artifacts/probe-<DATE>/`, because
the settled charge usually appears only *after* the pod is gone and cannot be collected from
inside it.

1. **Terminate the pod in the console. Confirm billing has stopped** — a killed process
   cannot stop its own meter, and the 2026-08-08 pod stayed allocated ~34 minutes after its
   run died.
2. Capture these into the **same** `probe-<DATE>/cost_evidence/` you created in §1, as
   **provider artifacts** — a screenshot or CSV/JSON export from the RunPod UI in each case.
   §1 already holds `01`–`03`; these are the closing half:

   | File | What it must show | Kind |
   |---|---|---|
   | `04_termination_confirmed.png` | pod state terminated/stopped, **with a visible timestamp** | provider |
   | `05_billing_stopped.png` | the billing or usage view showing the meter has stopped | provider |
   | `06_settled_charge.png` / `.csv` | the **settled** line item for this pod, once it appears | provider |
   | `07_elapsed_derived.txt` | pod create → terminate wall-clock, and the launcher's `elapsed=` from `PROBE_EXIT_RECORD` | **derived — not provider proof** |

3. Hash the folder once captured. The `cd` is fail-closed and the manifest excludes itself,
   so this is safe to re-run when the settled charge lands later:

This one runs on **your laptop**, where the login shell is zsh, so it is piped to `bash`
explicitly — an unmatched glob aborts a zsh command and zsh arrays are 1-indexed, and neither
failure would look like a failure here. Edit `<DATE>` inside the heredoc before pasting:

```bash
bash <<'SH'
set -o pipefail
DEST=~/Documents/llama-tools-artifacts/probe-<DATE>/cost_evidence
if ! cd "$DEST"; then
  echo "ABORT: $DEST does not exist — create it and re-capture; do NOT hash whatever"
  echo "       directory this shell happens to be sitting in."
else
  # 06 is deliberately absent from this list: the settled charge may not exist
  # yet. Every other slot must be present -- one file each, whatever extension.
  REQUIRED="01_rate_at_creation 02_image_selected 03_auto_terminate_set
            04_termination_confirmed 05_billing_stopped 07_elapsed_derived"
  BAD=(); N=0; SETTLED=no
  TMP=$(mktemp ./cost_evidence_sha256.XXXXXX) || BAD+=("could not create temp file")
  shopt -s nullglob
  for want in $REQUIRED; do
    matches=( "$want"* )
    if [[ "${#matches[@]}" -ne 1 ]]; then
      BAD+=("required evidence must have exactly one match: $want* (found ${#matches[@]})")
    elif [[ ! -f "${matches[0]}" || ! -s "${matches[0]}" ]]; then
      BAD+=("required evidence is non-regular or empty: ${matches[0]}")
    fi
  done
  settled_matches=( 06_settled_charge* )
  if [[ "${#settled_matches[@]}" -eq 1 \
        && -f "${settled_matches[0]}" && -s "${settled_matches[0]}" ]]; then
    SETTLED=yes
  elif [[ "${#settled_matches[@]}" -gt 0 ]]; then
    BAD+=("settled charge must have exactly one non-empty regular file when present")
  fi
  for f in *; do
    [[ -f "$f" ]] || continue
    case "$f" in cost_evidence_sha256.*) continue ;; esac   # never hash itself
    if [[ ! -s "$f" ]]; then
      BAD+=("empty evidence file: $f")
      continue
    fi
    shasum -a 256 "$f" >> "$TMP" || BAD+=("shasum failed (exit $?): $f")
    N=$((N+1))
  done
  if [[ "$SETTLED" == yes ]]; then STATE="settled charge PRESENT"
  else STATE="settled charge PENDING — cost stays 'unsettled, not attributable'"; fi
  if [[ "${#BAD[@]}" -eq 0 ]]; then
    mv -f "$TMP" cost_evidence_sha256.txt || BAD+=("could not write cost_evidence_sha256.txt")
  fi
  if [[ "${#BAD[@]}" -eq 0 ]]; then
    cat cost_evidence_sha256.txt
    echo "COST EVIDENCE COMPLETE ($N entries; $STATE)"
  else
    rm -f "$TMP"; printf 'ABORT: %s\n' "${BAD[@]}"
  fi
fi
SH
```

**A count is not a checklist.** The previous version accepted any one file as evidence, so a
folder holding only `07_elapsed_derived.txt` — the one row that is *not* provider proof —
printed a clean manifest. Each of `01`–`05` and `07` must now be present by name before the
token appears, with **exactly one non-empty regular file per slot**; zero-byte placeholders
and conflicting duplicate captures refuse. `06_settled_charge` stays optional **only** while
every cost statement remains explicitly unsettled; when present it has the same exactly-one,
non-empty rule. The token says which of those two worlds you are in rather than leaving it to
memory. Re-running after the charge lands re-hashes and flips `PENDING` to `PRESENT`.

**Capture timing is not a detail.** `01_rate_at_creation.png` and `03_auto_terminate_set.png`
belong to §1 **before Deploy** — after termination those views are gone, and a rate you can
no longer display is a rate you cannot evidence.

**The in-pod `--auto-terminate-set` attestation is operator-entered, not provider proof.** It
records what the operator said they set; it is not evidence of what was charged, and it does
not substitute for any row in that table. `07_elapsed_derived.txt` is likewise computed by
us, and is labelled so it can never be cited as a provider artifact.

**Until `settled_charge` exists, every cost statement about this run stays explicitly
unmeasured** — write "unsettled, not attributable" rather than a number. The 08-08 attempt
has an unsettled row whose value is identical to the mining pilot's receipt and **cannot be
attributed** to either run; repeating that mistake is how the $0.2131 → $0.2327 error
happened. A settled charge that never materialises is reported as absent, not estimated.

---

## Abort criteria — any one of these ends the run

**Every step gates on a positive token, not on the absence of an error.** If the expected
word did not appear, treat it as failure even when nothing looked wrong:

| Step | Must print | Absent ⇒ |
|---|---|---|
| §1 | `PRE-DEPLOY EVIDENCE READY` **before Deploy** | the rate and auto-terminate can never be evidenced again |
| §3 | `SOURCE PINNED` | do not run bootstrap — the source is not pinned to the reviewed SHA |
| §5 | `BOOTSTRAP COMPLETE` | stop, post the output, terminate |
| §6 | `round-trip OK` **and** `DEADLINES OK` | no deadlines exist; §7 will refuse |
| §7 | `LAUNCHED` | nothing started; nothing to monitor |
| §8 | `MONITOR ACTIVE` | the run is unmonitored — fix or terminate, do not walk away |
| §10 | `ACCEPTANCE PASS`, `acceptance exit: 0`, `MONITOR TERMINAL OK`, `ARTIFACTS COMPLETE (n entries)` | the numbers may not be reported (§0.5) |
| §11 | `COST EVIDENCE COMPLETE (n entries; …)` | the evidence set is incomplete — the cost of this run is not documented |

Plus:

- Bootstrap exits non-zero (any code).
- Any `ABORT:` line anywhere. Each one is enclosing — the side effect it guards does not run.
- The §6 round-tripped timestamp does not match the console by eye.
- Fewer than **45** minutes of work remain at §6 or at §7 (the deadlines are discarded
  automatically, and §7 re-checks the floor immediately before launching).
- Launcher exits **69** (gate failed) → take the §9 branch.
- Monitor reports **72** (died hard) → stop the pod, confirm billing stopped.
- Reproduction check returns anything other than **369/400**.
- Anything you did not expect. Post the output; do not improvise on a billing pod.
