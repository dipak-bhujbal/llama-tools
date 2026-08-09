#!/usr/bin/env bash
# Positive liveness monitor for a paid run.
#
# The monitor this replaces asserted nothing. It watched a file's mtime and
# warned when the file stopped growing, which is a proxy that is wrong in both
# directions: an 8B model load writes nothing for minutes while perfectly
# healthy (false alarm), and a process killed by the OOM killer leaves a log
# whose mtime is exactly as fresh as the moment it died (silence, at the one
# moment an alert was warranted). During the 2026-08-08 §0 probe a PID file
# existed while the process behind it did not, and nothing said so.
#
# So this asserts, in order, three positive facts:
#
#   1. The watched process is alive        -- kill -0 on the PID, or tmux has-session.
#   2. If it is NOT alive, a terminal record exists explaining why.
#   3. That record carries an exit code.
#
# and it writes a terminal status file so the answer survives the shell that
# produced it. The alertable condition is (1) false and (2) false: the process
# is gone and left no account of itself. That is a hard death -- SIGKILL, OOM
# killer, host preemption -- and it is the only thing worth waking someone for.
#
# Log staleness is still REPORTED, because it is useful context. It is never an
# alert. A quiet log with a live PID is a model loading; a quiet log with a dead
# PID is already caught by rule 1.

set -euo pipefail

readonly EXIT_ALIVE=0
readonly EXIT_USAGE=64
readonly EXIT_COMPLETED_OK=70          # process gone, terminal record says exit 0
readonly EXIT_COMPLETED_FAILED=71      # process gone, terminal record says nonzero
readonly EXIT_DIED_HARD=72             # process gone, NO terminal record -- the alert

# The footer strings written by scripts/launch_probe.sh's on_exit trap. If these
# are edited there they must be edited here; tests/test_probe_liveness.py greps
# both files to keep them from drifting apart.
readonly FOOTER_OK="RUN COMPLETE"
readonly FOOTER_FAIL="RUN DID NOT COMPLETE"
readonly EXIT_RECORD_PREFIX="PROBE_EXIT_RECORD"

# Substrings that mean the run is in trouble even while the process is still up.
# Reported, not fatal: the monitor never kills the thing it is watching, because
# a monitor that can terminate a paid run is a new way to lose one.
readonly ERROR_MARKERS=(
  "CUDA error"
  "illegal memory access"
  "out of memory"
  "CUDA out of memory"
  "Traceback (most recent call last)"
  "RuntimeError"
  "Killed"
  "Segmentation fault"
  "NCCL"
  "Xid"
)

usage() {
  cat >&2 <<'USAGE'
usage: probe_liveness.sh --log FILE --status-file FILE --pid N
                        [--tmux-session NAME] [--interval SECONDS] [--once]
                        [--allow-legacy-footer]

  --log             stdout/stderr log of the watched run (scanned for the EXIT footer)
  --status-file     JSON terminal status is written here, atomically
  --pid             PID of the launcher. REQUIRED, and the sole liveness authority.
  --tmux-session    tmux session name. Recorded as context ONLY -- never used to
                    decide whether the run is alive. See the note below.
  --interval        seconds between checks in loop mode (default 30)
  --once            perform one assertion pass and exit with the state code
  --allow-legacy-footer
                    accept a prose "RUN COMPLETE"/"RUN DID NOT COMPLETE" footer that
                    carries no PID. Off by default: such a footer cannot be attributed
                    to this run, so a log appended by two consecutive runs would let
                    the earlier run's outcome be reported as this one's.

Why tmux is not a liveness signal: the runbook has the operator run this monitor
from a second pane of the same session. The session therefore outlives the
launcher by construction, so treating "session exists" as "run is alive" would
report RUNNING forever after the launcher dies -- the one event this exists to
catch.

exit codes:
  0   watched process is alive
  64  usage error
  70  process gone; authenticated terminal record reports exit 0
  71  process gone; authenticated terminal record reports a nonzero exit
  72  process gone with no authenticated terminal record -- hard death, the alert
USAGE
}

log_file=""
status_file=""
watch_pid=""
tmux_session=""
interval=30
once=0
allow_legacy_footer=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log)           log_file="${2:-}"; shift 2 ;;
    --status-file)   status_file="${2:-}"; shift 2 ;;
    --pid)           watch_pid="${2:-}"; shift 2 ;;
    --tmux-session)  tmux_session="${2:-}"; shift 2 ;;
    --interval)      interval="${2:-}"; shift 2 ;;
    --once)          once=1; shift ;;
    --allow-legacy-footer) allow_legacy_footer=1; shift ;;
    -h|--help)       usage; exit "${EXIT_USAGE}" ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage; exit "${EXIT_USAGE}" ;;
  esac
done

missing=()
[[ -z "${log_file}" ]] && missing+=("--log")
[[ -z "${status_file}" ]] && missing+=("--status-file")
[[ -z "${watch_pid}" ]] && missing+=("--pid")
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "ERROR: missing required: ${missing[*]}" >&2
  usage
  exit "${EXIT_USAGE}"
fi
if [[ -n "${watch_pid}" && ! "${watch_pid}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --pid must be numeric, got: ${watch_pid}" >&2
  exit "${EXIT_USAGE}"
fi
if [[ ! "${interval}" =~ ^[0-9]+$ ]] || [[ "${interval}" -lt 1 ]]; then
  echo "ERROR: --interval must be a positive integer, got: ${interval}" >&2
  exit "${EXIT_USAGE}"
fi

# ---------------------------------------------------------------------------
# Assertion 1: is the process alive?
# ---------------------------------------------------------------------------
# `kill -0` answers "does a process with this PID exist and may I signal it",
# which is the question. A PID *file* answers nothing -- it is a claim written
# once, by a process that cannot update it after dying.
pid_alive() {
  [[ -n "${watch_pid}" ]] || return 1
  kill -0 "${watch_pid}" 2>/dev/null
}

tmux_alive() {
  [[ -n "${tmux_session}" ]] || return 1
  command -v tmux >/dev/null 2>&1 || return 1
  tmux has-session -t "${tmux_session}" 2>/dev/null
}

# The PID is the sole authority. tmux_alive is collected for the status file as
# context, and deliberately NOT consulted here.
#
# An earlier version returned `pid_alive || tmux_alive`. Combined with the
# runbook's own instruction to run this monitor from a second pane of the
# watched session, that OR guaranteed the monitor would report RUNNING forever
# after the launcher died: the session was being kept alive by the monitor
# itself. A liveness check whose own presence satisfies its condition is not a
# check.
process_alive() {
  pid_alive
}

# ---------------------------------------------------------------------------
# Assertion 2/3: does a terminal record exist, and what exit code does it carry?
# ---------------------------------------------------------------------------
# Sets: footer_state (complete|failed|absent|foreign) and footer_exit.
#
# `foreign` means a terminal record was found but it belongs to a DIFFERENT pid
# than the one being watched — an earlier run appending to the same log. It is
# deliberately not treated as a terminal record for this run: reporting another
# process's clean exit as ours is exactly the stale-evidence failure this
# monitor exists to prevent, and it fails closed to the hard-death alert.
scan_footer() {
  footer_state="absent"
  footer_exit=""
  [[ -r "${log_file}" ]] || return 0

  # Read the tail only. A long run's log is large and the footer is the last
  # thing written; scanning the whole file every interval is work for nothing.
  local tail_text record record_pid
  tail_text="$(tail -n 200 "${log_file}" 2>/dev/null || true)"

  # Preferred: the machine-readable record written by launch_probe.sh's trap.
  record="$(grep -F "${EXIT_RECORD_PREFIX}" <<<"${tail_text}" | tail -n 1 || true)"
  if [[ -n "${record}" ]]; then
    record_pid="$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' <<<"${record}")"
    # A record with no parseable pid authenticates nothing. The previous version
    # required `-n record_pid` before it would even consider a mismatch, so a
    # truncated or malformed line like `PROBE_EXIT_RECORD exit=0` fell straight
    # through to "clean completion" — an unauthenticated record accepted in the
    # reassuring direction, which is the only direction that actually costs
    # anything. Missing pid is now treated as no record at all.
    if [[ -z "${record_pid}" ]]; then
      footer_state="malformed_record"
      return 0
    fi
    if [[ "${record_pid}" != "${watch_pid}" ]]; then
      footer_state="foreign"
      return 0
    fi
    footer_exit="$(sed -n 's/.* exit=\([0-9][0-9]*\).*/\1/p' <<<"${record}")"
    if [[ "${footer_exit}" == "0" ]]; then
      footer_state="complete"
    elif [[ -n "${footer_exit}" ]]; then
      footer_state="failed"
    fi
    return 0
  fi

  # Fallback: the human footer, from a launcher predating the record line. It
  # carries no pid, so it cannot be attributed to this run — an old footer left
  # in a reused log would be accepted as this run's outcome, which is the same
  # stale-evidence hole the pid check above closes. Off unless the operator
  # explicitly opts in.
  if [[ "${allow_legacy_footer}" -ne 1 ]]; then
    if grep -qE "${FOOTER_FAIL}|${FOOTER_OK}" <<<"${tail_text}"; then
      footer_state="unauthenticated"
    fi
    return 0
  fi

  if grep -qF "${FOOTER_FAIL}" <<<"${tail_text}"; then
    footer_state="failed"
    # Footer form: "RUN DID NOT COMPLETE — exit 68, elapsed 64s"
    footer_exit="$(grep -F "${FOOTER_FAIL}" <<<"${tail_text}" | tail -n 1 \
      | sed -n 's/.*exit \([0-9][0-9]*\).*/\1/p')"
  elif grep -qF "${FOOTER_OK}" <<<"${tail_text}"; then
    footer_state="complete"
    footer_exit="0"
  fi
  return 0
}

# Error markers seen anywhere in the log, written to a sidecar rather than
# embedded in JSON -- arbitrary log text through a hand-rolled JSON escaper is
# a bug waiting to happen, and a truncated status file is worse than a verbose one.
# A SCAN THAT DID NOT RUN IS NOT A SCAN THAT FOUND NOTHING.
#
# Every grep here used to carry `2>/dev/null`, and grep's three exit codes were
# collapsed into two: 0 meant found, and *everything else* meant not-found. But
# grep exits 1 for "no match" and >=2 for "I could not read that" — an I/O
# error, a vanished file, a permissions change mid-run. Under the old code the
# second case reported `error markers: 0` and `error_markers_seen: 0`, i.e. a
# clean bill of health issued by a check that never completed. That is the exact
# shape of the Xid-regex defect fixed in 8659fb0.
#
# It could not suppress the DIED HARD alert — that verdict comes from
# process_alive and footer_state, not from here — so the blast radius is the
# reporting surface. On a paid run the reporting surface is what a human reads
# at 2am to decide whether to keep spending, which is not a small thing to lie
# on. `marker_scan` now carries ok | unreadable | failed, and marker_count is
# reported as null rather than 0 whenever the scan did not actually complete.
scan_error_markers() {
  marker_count=0
  marker_names=""
  marker_scan="ok"
  marker_scan_detail=""

  if [[ ! -r "${log_file}" ]]; then
    marker_scan="unreadable"
    marker_scan_detail="log file is absent or not readable: ${log_file}"
    return 0
  fi

  local marker status err
  for marker in "${ERROR_MARKERS[@]}"; do
    status=0
    # stderr is CAPTURED, not discarded, so a real failure can be reported
    # rather than mistaken for "no match".
    err="$(grep -qF -e "${marker}" "${log_file}" 2>&1)" || status=$?
    case "${status}" in
      0)
        marker_count=$(( marker_count + 1 ))
        marker_names="${marker_names:+${marker_names}, }${marker}"
        ;;
      1)
        : # genuinely absent from the log. Not an error, and must never become one.
        ;;
      *)
        marker_scan="failed"
        marker_scan_detail="grep exited ${status} scanning for '${marker}': ${err}"
        return 0
        ;;
    esac
  done

  if [[ "${marker_count}" -gt 0 ]]; then
    # Context extraction. `|| true` is retained ONLY here and only because a
    # non-zero exit is expected: this re-greps markers already known present,
    # so the pipeline's status reflects `tail`, not discovery. A failure to
    # write context does not invalidate the counts above.
    : > "${status_file}.errors.txt"
    for marker in "${ERROR_MARKERS[@]}"; do
      grep -nF -e "${marker}" "${log_file}" 2>>"${status_file}.errors.txt" \
        | tail -n 40 >> "${status_file}.errors.txt" || true
    done
  fi
  return 0
}

log_age_seconds() {
  if [[ ! -e "${log_file}" ]]; then
    echo ""
    return 0
  fi
  local mtime now
  # BSD (macOS) and GNU stat take different flags; the pod is Linux, a reviewer's
  # laptop may not be.
  mtime="$(stat -f %m "${log_file}" 2>/dev/null || stat -c %Y "${log_file}" 2>/dev/null || echo "")"
  [[ -z "${mtime}" ]] && { echo ""; return 0; }
  now="$(date -u +%s)"
  echo $(( now - mtime ))
}

# ---------------------------------------------------------------------------
# JSON string escaping.
#
# Every dynamic string in the status file used to be interpolated raw with
# `printf '"%s"'`. A log path containing a backslash — `/tmp/a\qb` — produced
# `Invalid escape` and made the whole artifact unparseable: `jq empty` exited 5
# while the monitor itself exited 72 believing it had written a clean record.
# The status file is the durable output of a monitor watching a paid run, so it
# has to survive exactly the inputs that show up when things are going wrong:
# odd paths, quoted messages, and multi-line diagnostics from a failed scan.
#
# One prior version "handled" this by replacing `"` with `'` in a single field,
# which mangles the data and still leaves backslashes and newlines broken.
json_escape() {
  local s="$1" out="" i c code
  s="${s//\\/\\\\}"      # backslash FIRST, or it doubles the escapes added below
  s="${s//\"/\\\"}"
  s="${s//$'\b'/\\b}"
  s="${s//$'\f'/\\f}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\t'/\\t}"
  # Any remaining C0 control character has no short form and must be \u00XX,
  # or the artifact is invalid JSON for a reason nobody will guess.
  #
  # Do NOT express this as `[$'\x01'-$'\x1f']`. Bash interprets a bracket
  # range using the active locale's collation order, not numeric byte order.
  # Under en_US.UTF-8 that range omitted vertical tab (U+000B), so the same
  # input produced valid JSON under LC_ALL=C and invalid JSON on a pod whose
  # locale was never pinned. Numeric comparison makes the property independent
  # of locale. Bash strings cannot contain NUL, hence the lower bound is 1.
  for (( i=0; i<${#s}; i++ )); do
    c="${s:i:1}"
    printf -v code '%d' "'${c}"
    if (( code >= 1 && code <= 31 )); then
      printf -v c '\\u%04x' "${code}"
    fi
    out+="${c}"
  done
  s="${out}"
  printf '%s' "${s}"
}

# Atomic write: a monitor that is itself killed mid-write must not leave a
# half-parsed status file behind, because the next reader would draw a
# conclusion from a truncated record.
write_status() {
  local state="$1" exit_code="$2" alert="$3" detail="$4"
  local age tmp
  age="$(log_age_seconds)"
  tmp="${status_file}.tmp.$$"
  {
    printf '{\n'
    printf '  "schema": "probe_liveness/v1",\n'
    printf '  "checked_at_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "state": "%s",\n' "$(json_escape "${state}")"
    if [[ -n "${exit_code}" ]]; then
      printf '  "exit_code": %s,\n' "${exit_code}"
    else
      printf '  "exit_code": null,\n'
    fi
    printf '  "alert": %s,\n' "${alert}"
    printf '  "watched_pid": %s,\n' "${watch_pid:-null}"
    printf '  "pid_alive": %s,\n' "$(pid_alive && echo true || echo false)"
    if [[ -n "${tmux_session}" ]]; then
      printf '  "tmux_session": "%s",\n' "$(json_escape "${tmux_session}")"
      printf '  "tmux_alive": %s,\n' "$(tmux_alive && echo true || echo false)"
    else
      printf '  "tmux_session": null,\n'
      printf '  "tmux_alive": null,\n'
    fi
    printf '  "footer_state": "%s",\n' "$(json_escape "${footer_state}")"
    # null, not 0, when the scan did not complete. A reader cannot distinguish
    # "scanned, found nothing" from "never scanned" if both render as 0, and
    # only one of those is reassuring.
    if [[ "${marker_scan}" == "ok" ]]; then
      printf '  "error_markers_seen": %s,\n' "${marker_count}"
    else
      printf '  "error_markers_seen": null,\n'
    fi
    printf '  "error_marker_scan": "%s",\n' "$(json_escape "${marker_scan}")"
    printf '  "error_marker_scan_detail": "%s",\n' "$(json_escape "${marker_scan_detail}")"
    printf '  "error_marker_names": "%s",\n' "$(json_escape "${marker_names}")"
    printf '  "log_file": "%s",\n' "$(json_escape "${log_file}")"
    # Informational only. This field must never drive an alert: see the header.
    if [[ -n "${age}" ]]; then
      printf '  "log_age_seconds": %s,\n' "${age}"
    else
      printf '  "log_age_seconds": null,\n'
    fi
    printf '  "log_age_is_not_an_alert_signal": true,\n'
    printf '  "detail": "%s"\n' "$(json_escape "${detail}")"
    printf '}\n'
  } > "${tmp}"
  mv -f "${tmp}" "${status_file}"
}

check_once() {
  scan_footer
  scan_error_markers

  if process_alive; then
    # Only the pid appears as the reason. tmux is appended as context and
    # labelled as such, so nobody reading this line later concludes the session
    # check contributed to the verdict.
    local how="pid ${watch_pid} alive"
    if tmux_alive; then
      how="${how} (context: tmux session ${tmux_session} also present)"
    fi
    write_status "running" "" "false" "${how}"
    if [[ "${marker_scan}" == "ok" ]]; then
      echo "[liveness] RUNNING — ${how}; error markers: ${marker_count}"
    else
      echo "[liveness] RUNNING — ${how}; error-marker scan ${marker_scan^^}:" \
           "${marker_scan_detail}" >&2
    fi
    return "${EXIT_ALIVE}"
  fi

  # Process is gone. Rule 2: is there an account of why?
  case "${footer_state}" in
    complete)
      write_status "exited" "${footer_exit}" "false" "process gone; EXIT footer reports clean completion"
      echo "[liveness] EXITED 0 — run completed and wrote its footer."
      return "${EXIT_COMPLETED_OK}"
      ;;
    failed)
      write_status "exited" "${footer_exit}" "true" "process gone; EXIT footer reports a nonzero exit"
      echo "[liveness] EXITED ${footer_exit:-unknown} — run failed but wrote its footer."
      return "${EXIT_COMPLETED_FAILED}"
      ;;
    foreign)
      write_status "died_hard" "" "true" \
        "process gone; the only terminal record in this log belongs to a different pid, so it is not evidence about this run"
      echo "[liveness] ALERT: DIED HARD — process gone; the terminal record in" >&2
      echo "[liveness] this log carries a different pid, so it describes an" >&2
      echo "[liveness] earlier run appending to the same file, not this one." >&2
      echo "[liveness] STOP THE POD AND CONFIRM BILLING STOPPED — a dead process" >&2
      echo "[liveness] does not stop its own meter." >&2
      return "${EXIT_DIED_HARD}"
      ;;
    malformed_record)
      write_status "died_hard" "" "true" \
        "process gone; a PROBE_EXIT_RECORD line is present but carries no parseable pid, so it authenticates nothing"
      echo "[liveness] ALERT: DIED HARD — process gone. A PROBE_EXIT_RECORD line" >&2
      echo "[liveness] is present but carries no parseable pid, so it cannot be" >&2
      echo "[liveness] attributed to this run. --allow-legacy-footer does NOT" >&2
      echo "[liveness] apply: this is a corrupt record, not an old-format log." >&2
      echo "[liveness] Most likely the log was truncated mid-write." >&2
      echo "[liveness] STOP THE POD AND CONFIRM BILLING STOPPED — a dead process" >&2
      echo "[liveness] does not stop its own meter." >&2
      return "${EXIT_DIED_HARD}"
      ;;
    unauthenticated)
      write_status "died_hard" "" "true" \
        "process gone; the log has a prose footer but no pid-bearing record, so it cannot be attributed to this run (pass --allow-legacy-footer to accept it)"
      echo "[liveness] ALERT: DIED HARD — process gone. The log has a prose EXIT" >&2
      echo "[liveness] footer but no pid-bearing PROBE_EXIT_RECORD, so it cannot" >&2
      echo "[liveness] be attributed to this run. If this log is from a launcher" >&2
      echo "[liveness] predating that record, re-run with --allow-legacy-footer." >&2
      echo "[liveness] STOP THE POD AND CONFIRM BILLING STOPPED — a dead process" >&2
      echo "[liveness] does not stop its own meter." >&2
      return "${EXIT_DIED_HARD}"
      ;;
    *)
      write_status "died_hard" "" "true" \
        "process gone and NO EXIT footer in the log: the launcher never ran its trap"
      echo "[liveness] ALERT: DIED HARD — process gone, no EXIT footer." >&2
      echo "[liveness] The launcher's exit trap never ran, so this was not an" >&2
      echo "[liveness] orderly exit: SIGKILL, the OOM killer, or host preemption." >&2
      echo "[liveness] STOP THE POD AND CONFIRM BILLING STOPPED — a dead process" >&2
      echo "[liveness] does not stop its own meter." >&2
      return "${EXIT_DIED_HARD}"
      ;;
  esac
}

# check_once returns a VERDICT, not a success/failure status: EXIT_ALIVE and the
# DIED_HARD codes are both non-zero and both expected. That is why this used to
# open a `set +e` window. `|| rc=$?` captures the same value without one, so
# `set -euo pipefail` holds for every line of this file rather than for most of
# them -- the same conversion applied to bootstrap_pod.sh and launch_probe.sh.
if [[ "${once}" -eq 1 ]]; then
  rc=0
  check_once || rc=$?
  exit "${rc}"
fi

echo "[liveness] watching ${watch_pid:+pid ${watch_pid}}${tmux_session:+ tmux ${tmux_session}} every ${interval}s"
echo "[liveness] status file: ${status_file}"
while true; do
  rc=0
  check_once || rc=$?
  if [[ "${rc}" -ne "${EXIT_ALIVE}" ]]; then
    exit "${rc}"
  fi
  sleep "${interval}"
done
