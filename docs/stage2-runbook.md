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

**The clone is pinned to the reviewed SHA _before_ anything from it is executed.** A clone
of the default branch is a moving target: `main` advancing between review and run would
change the behaviour of the very script that enforces exact-SHA execution, while the runbook
still claimed the reviewed commit ran.

```bash
cd /workspace
export RUNPOD_IMAGE_NAME='<IMAGE_TAG>'
export HF_HOME=/root/.cache/huggingface
mkdir -p "$HF_HOME" /workspace/persist/study2

rm -rf /workspace/src        # so a re-paste after a typo cannot clone into a dirty tree
git clone https://github.com/dipak-bhujbal/llama-tools.git src
git -C src checkout --detach 08be5d3b0c1293ac229fe7b04e2931f14bd149d0

HEAD_NOW=$(git -C src rev-parse HEAD 2>/dev/null)
TREE_DIRT=$(git -C src status --porcelain 2>/dev/null)
if [[ "$HEAD_NOW" != "08be5d3b0c1293ac229fe7b04e2931f14bd149d0" ]]; then
  echo "ABORT: src HEAD is '$HEAD_NOW', expected 08be5d3b0c1293ac229fe7b04e2931f14bd149d0"
elif [[ -n "$TREE_DIRT" ]]; then
  echo "ABORT: src working tree is not clean:"; echo "$TREE_DIRT"
else
  git -C src bundle create /workspace/llama-tools.bundle --all
  sha256sum /workspace/llama-tools.bundle | awk '{print $1}' > /workspace/llama-tools.bundle.sha256
  git -C src remote get-url origin > /workspace/persist/study2/clone_source_url.txt
  printf '%s\n' "$HEAD_NOW"        > /workspace/persist/study2/clone_detached_head.txt
  echo "SOURCE PINNED at $HEAD_NOW, tree clean, bundle written"
fi
```

**Proceed only on `SOURCE PINNED`.** The bundle is created *after* the detach, so nothing is
packaged from an unpinned tree; `--all` still carries every ref, so the bundle remains a
complete mirror. The recorded URL is `git remote get-url origin` — what the clone actually
came from — not the string typed above, which would agree with itself even if the clone had
come from somewhere else.

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
if [[ "$(git -C /workspace/src rev-parse HEAD 2>/dev/null)" != "08be5d3b0c1293ac229fe7b04e2931f14bd149d0" ]]; then
  echo "ABORT: /workspace/src is not at the reviewed SHA — redo §3, do not run bootstrap"
elif [[ -n "$(git -C /workspace/src status --porcelain 2>/dev/null)" ]]; then
  echo "ABORT: /workspace/src is dirty — redo §3, do not run bootstrap"
else
  bash src/scripts/bootstrap_pod.sh \
    --bundle /workspace/llama-tools.bundle \
    --bundle-sha256-file /workspace/llama-tools.bundle.sha256 \
    --commit 08be5d3b0c1293ac229fe7b04e2931f14bd149d0 \
    --out-root /workspace/persist/study2 \
    --auto-terminate-set '<TERMINATE_UTC>@<RATE>'
fi
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
    if [[ "$MINS_LEFT" -lt 40 ]]; then
      unset PROVIDER_EPOCH DEADLINE_EPOCH
      echo "ABORT: only $MINS_LEFT min of work left, need >= 40 — deadlines discarded"
    else
      echo "DEADLINES OK"
    fi
  fi
fi
```

**Proceed only if you saw `round-trip OK` and `DEADLINES OK`, and no `ABORT` line.** Every
failure path — bad shape, unparseable, round-trip mismatch, **and too little time left** —
leaves `PROVIDER_EPOCH` and `DEADLINE_EPOCH` unset, and §7 mechanically refuses to launch
without both. The guard is not the `ABORT` message; the guard is that the values a launch
needs do not exist. The round-trip is what proves the epoch means the instant the console
displayed: compare the printed `round-trip OK` value against the console by eye.

**Why the ≥ 40 minute gate:** the launcher already refuses to start if its deadline is not
strictly earlier than the provider's, and refuses a deadline already past. Neither catches
the expensive case — enough time to *start* but not to *finish*. Launching into a run that
gets killed mid-generation is billed in full and yields nothing.

---

## 7. Launch — ladder gates the probe, same pod

**The launcher must run as a child of _this_ shell — not in a new tmux session.** This is
not a style preference; a second `tmux new-session` would break the run. See the box below.

**The guards below enclose the launch.** They do not warn and fall through — if any
precondition is missing, the `nohup` line is never reached, because it lives inside the
`else`. Nothing is deleted and nothing is spent on a failed check.

```bash
cd /workspace/llama-tools
unset LAUNCHER_BG_PID

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ABORT: HF_TOKEN not set in this shell — redo §4. Not launching."
elif [[ -z "${PROVIDER_EPOCH:-}" || -z "${DEADLINE_EPOCH:-}" ]]; then
  echo "ABORT: §6 did not complete — no verified deadlines. Not launching."
elif [[ "$DEADLINE_EPOCH" -ge "$PROVIDER_EPOCH" ]]; then
  echo "ABORT: launcher deadline is not strictly earlier than the provider's. Not launching."
elif [[ "$DEADLINE_EPOCH" -le "$(date -u +%s)" ]]; then
  echo "ABORT: launcher deadline has already passed. Not launching."
else
  # Only now, past every guard: clear any pid from an earlier attempt and launch.
  rm -f /workspace/persist/study2/launcher.pid
  nohup bash scripts/launch_probe.sh \
    --commit 08be5d3b0c1293ac229fe7b04e2931f14bd149d0 \
    --provider-deadline-epoch "$PROVIDER_EPOCH" \
    --deadline-epoch "$DEADLINE_EPOCH" \
    --out-root /workspace/persist/study2 \
    > /workspace/persist/study2/probe.log 2>&1 &
  LAUNCHER_BG_PID=$!
  echo "LAUNCHED; shell-visible pid $LAUNCHER_BG_PID"
fi
```

**Proceed to §8 only on `LAUNCHED`.** `LAUNCHER_BG_PID` is unset first, so if the launch did
not happen there is no pid for §8 to compare against and §8 stops too. The last two guards
duplicate checks the launcher already makes internally — deliberately, so the failure costs
a shell round-trip instead of a process start on a billing pod.

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
| **1** | raw base, explicit `cuda:0`, no `device_map` | Excludes PEFT, placement and adapter state — none is present. Implicates **the card, the driver, the torch/CUDA build, or the base weights at this revision** | **Terminate. Retry once on a second node** — that discriminates host from the rest, it does not by itself convict the host. Reproduces on two nodes ⇒ not the card: pin a known-good Transformers/Torch pair for §0 only, at the **image boundary** (exact image tag + a bootstrap runtime-equality assertion), not via a torch overlay in `requirements-probe.txt`. The pinned base-weights revision stays an open cause until the retry rules the host out |
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
python3 - <<'PY'
import collections, json, pathlib, sys

ROOT = pathlib.Path("/workspace/persist/study2")
EXPECT = {"study2_probe_multiple": 400, "study2_probe_simple_python": 800}
REPRO_DIR, REPRO_CAND, REPRO_WANT, REPRO_DENOM = "study2_probe_simple_python", "sft", 369, 400

fail = []
def check(cond, msg):
    if not cond:
        fail.append(msg)

for d, expected_rows in EXPECT.items():
    run = ROOT / d
    man_p, gen_p = run / "run_manifest.json", run / "generations.jsonl"
    if not man_p.exists() or not gen_p.exists():
        fail.append(f"{d}: MISSING " + ", ".join(
            p.name for p in (man_p, gen_p) if not p.exists()))
        continue
    m = json.loads(man_p.read_text())
    check(m.get("status") == "complete",
          f"{d}: status={m.get('status')!r}, expected 'complete'")
    check(m.get("expected_rows") == expected_rows,
          f"{d}: manifest expected_rows={m.get('expected_rows')}, prereg says {expected_rows}")
    check(m.get("rows_written") == m.get("expected_rows"),
          f"{d}: rows_written={m.get('rows_written')} != expected_rows={m.get('expected_rows')}")
    v = m.get("validation") or {}
    check(v.get("ok") is True, f"{d}: validation.ok={v.get('ok')!r}")
    check(v.get("unparseable_lines") == 0,
          f"{d}: validation.unparseable_lines={v.get('unparseable_lines')!r}")
    for k in ("duplicate_pairs", "missing_pairs", "extra_pairs"):
        check(not v.get(k), f"{d}: validation.{k} non-empty: {v.get(k)!r}")

    rows = [json.loads(l) for l in gen_p.read_text().splitlines() if l.strip()]
    check(len(rows) == expected_rows,
          f"{d}: {len(rows)} rows on disk, expected {expected_rows}")
    pairs = collections.Counter((r["id"], r["model_name"]) for r in rows)
    dupes = [p for p, c in pairs.items() if c > 1]
    check(not dupes, f"{d}: {len(dupes)} duplicated (id,candidate) pairs, e.g. {dupes[:3]}")
    check(len(pairs) == expected_rows,
          f"{d}: {len(pairs)} unique (id,candidate) pairs, expected {expected_rows}")
    cands = sorted({r["model_name"] for r in rows})
    check(cands == ["base", "sft"], f"{d}: candidates {cands}, expected ['base', 'sft']")

gen_p = ROOT / REPRO_DIR / "generations.jsonl"
if not gen_p.exists():
    fail.append(f"reproduction gate: {REPRO_DIR}/generations.jsonl missing")
else:
    sub = [json.loads(l) for l in gen_p.read_text().splitlines() if l.strip()]
    sub = [r for r in sub if r["model_name"] == REPRO_CAND]
    got = sum(1 for r in sub if r.get("overall_ok"))
    print(f"reproduction: {REPRO_CAND} on {REPRO_DIR} scored {got}/{len(sub)} "
          f"(prereg §0.4 expects {REPRO_WANT}/{REPRO_DENOM})")
    check(len(sub) == REPRO_DENOM,
          f"reproduction gate: {len(sub)} {REPRO_CAND} rows, expected {REPRO_DENOM}")
    check(got == REPRO_WANT,
          f"REPRODUCTION GATE FAILED: scored {got}, prereg §0.4 expects {REPRO_WANT}")

if fail:
    print("\n".join("FAIL: " + f for f in fail))
    print(f"\n{len(fail)} FAILURE(S) — these numbers may not be reported (§0.5).")
    sys.exit(1)
print("ACCEPTANCE PASS: both runs complete, counts and coverage exact, reproduction gate met.")
PY
echo "acceptance exit: $?"
```

**`acceptance exit: 0` and `ACCEPTANCE PASS` are the only results that let these numbers be
reported.** A run that does not write exactly `n_prompts × n_candidates` rows is
**incomplete** under §0.5; unique-pair coverage is checked separately from the row count
because a duplicate can mask a missing pair while the total still looks right. The
reproduction figure is **computed from `overall_ok` and compared**, not asserted — any value
other than 369/400 is stop-and-report, not something to reconcile afterwards.

Then hash the required artifacts — **by explicit list, failing loudly on absence.** A glob
with `2>/dev/null` cannot distinguish "hashed everything" from "matched nothing":

```bash
cd /workspace/persist/study2
{ for f in study2_probe_multiple/generations.jsonl      study2_probe_multiple/run_manifest.json \
           study2_probe_simple_python/generations.jsonl study2_probe_simple_python/run_manifest.json \
           isolation_ladder/isolation_ladder.json       probe.log \
           clone_source_url.txt                         clone_detached_head.txt; do
    if [[ -s "$f" ]]; then sha256sum "$f"; else echo "MISSING OR EMPTY: $f"; fi
  done
} > artifact_sha256.txt
cat artifact_sha256.txt
if grep -q 'MISSING OR EMPTY' artifact_sha256.txt; then
  echo "ABORT: required artifacts missing — listed above"
else
  echo "ARTIFACTS COMPLETE"
fi
```

The verdict is read back out of the file rather than carried in a shell variable: the loop
would otherwise run in a pipeline subshell, and a flag set there never reaches the shell that
tests it — the same class of silent-pass bug as finding 1.

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
2. Capture these into `probe-<DATE>/cost_evidence/`, as **provider artifacts, not typed
   notes** — a screenshot or CSV/JSON export from the RunPod UI in each case:

   | File | What it must show |
   |---|---|
   | `rate_at_creation.png` | the pod's card and **$/hr as the console displayed it** |
   | `termination_confirmed.png` | pod state terminated/stopped, **with a visible timestamp** |
   | `billing_stopped.png` | the billing or usage view showing the meter has stopped |
   | `settled_charge.png` / `.csv` | the **settled** line item for this pod, once it appears |
   | `elapsed.txt` | pod create → terminate wall-clock, and the launcher's `elapsed=` from `PROBE_EXIT_RECORD` |

3. Hash the folder once captured, so the evidence is fixed at collection time:

```bash
cd ~/Documents/llama-tools-artifacts/probe-<DATE>/cost_evidence
shasum -a 256 * > cost_evidence_sha256.txt && cat cost_evidence_sha256.txt
```

**The in-pod `--auto-terminate-set` attestation is operator-entered, not provider proof.** It
records what the operator said they set; it is not evidence of what was charged, and it does
not substitute for any row in that table.

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
| §3 | `SOURCE PINNED` | do not run bootstrap — the source is not pinned to the reviewed SHA |
| §5 | `BOOTSTRAP COMPLETE` | stop, post the output, terminate |
| §6 | `round-trip OK` **and** `DEADLINES OK` | no deadlines exist; §7 will refuse |
| §7 | `LAUNCHED` | nothing started; nothing to monitor |
| §8 | `MONITOR ACTIVE` | the run is unmonitored — fix or terminate, do not walk away |
| §10 | `ACCEPTANCE PASS`, `acceptance exit: 0`, `ARTIFACTS COMPLETE` | the numbers may not be reported (§0.5) |

Plus:

- Bootstrap exits non-zero (any code).
- Any `ABORT:` line anywhere. Each one is enclosing — the side effect it guards does not run.
- The §6 round-tripped timestamp does not match the console by eye.
- Fewer than 40 minutes of work remain at §6 (the deadlines are discarded automatically).
- Launcher exits **69** (gate failed) → take the §9 branch.
- Monitor reports **72** (died hard) → stop the pod, confirm billing stopped.
- Reproduction check returns anything other than **369/400**.
- Anything you did not expect. Post the output; do not improvise on a billing pod.
