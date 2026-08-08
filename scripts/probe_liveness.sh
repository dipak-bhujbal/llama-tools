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
usage: probe_liveness.sh --log FILE --status-file FILE
                        (--pid N | --tmux-session NAME)
                        [--interval SECONDS] [--once]

  --log            stdout/stderr log of the watched run (scanned for the EXIT footer)
  --status-file    JSON terminal status is written here, atomically
  --pid            PID of the launcher to assert alive
  --tmux-session   tmux session name to assert alive (use instead of, or with, --pid)
  --interval       seconds between checks in loop mode (default 30)
  --once           perform one assertion pass and exit with the state code

exit codes:
  0   watched process is alive
  64  usage error
  70  process gone; terminal record reports exit 0
  71  process gone; terminal record reports a nonzero exit
  72  process gone with NO terminal record -- hard death, this is the alert
USAGE
}

log_file=""
status_file=""
watch_pid=""
tmux_session=""
interval=30
once=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log)           log_file="${2:-}"; shift 2 ;;
    --status-file)   status_file="${2:-}"; shift 2 ;;
    --pid)           watch_pid="${2:-}"; shift 2 ;;
    --tmux-session)  tmux_session="${2:-}"; shift 2 ;;
    --interval)      interval="${2:-}"; shift 2 ;;
    --once)          once=1; shift ;;
    -h|--help)       usage; exit "${EXIT_USAGE}" ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage; exit "${EXIT_USAGE}" ;;
  esac
done

missing=()
[[ -z "${log_file}" ]] && missing+=("--log")
[[ -z "${status_file}" ]] && missing+=("--status-file")
if [[ -z "${watch_pid}" && -z "${tmux_session}" ]]; then
  missing+=("--pid or --tmux-session")
fi
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

process_alive() {
  pid_alive || tmux_alive
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
    if [[ -n "${watch_pid}" && -n "${record_pid}" && "${record_pid}" != "${watch_pid}" ]]; then
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

  # Fallback: the human footer, for logs from a launcher predating the record
  # line. No pid to check against, so this is weaker evidence by construction.
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
scan_error_markers() {
  marker_count=0
  marker_names=""
  [[ -r "${log_file}" ]] || return 0
  local marker
  for marker in "${ERROR_MARKERS[@]}"; do
    if grep -qF "${marker}" "${log_file}" 2>/dev/null; then
      marker_count=$(( marker_count + 1 ))
      marker_names="${marker_names:+${marker_names}, }${marker}"
    fi
  done
  if [[ "${marker_count}" -gt 0 ]]; then
    grep -nF -e "${ERROR_MARKERS[0]}" "${log_file}" 2>/dev/null | tail -n 40 \
      > "${status_file}.errors.txt" || true
    for marker in "${ERROR_MARKERS[@]:1}"; do
      grep -nF -e "${marker}" "${log_file}" 2>/dev/null | tail -n 40 \
        >> "${status_file}.errors.txt" || true
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
    printf '  "state": "%s",\n' "${state}"
    if [[ -n "${exit_code}" ]]; then
      printf '  "exit_code": %s,\n' "${exit_code}"
    else
      printf '  "exit_code": null,\n'
    fi
    printf '  "alert": %s,\n' "${alert}"
    printf '  "watched_pid": %s,\n' "${watch_pid:-null}"
    printf '  "pid_alive": %s,\n' "$(pid_alive && echo true || echo false)"
    if [[ -n "${tmux_session}" ]]; then
      printf '  "tmux_session": "%s",\n' "${tmux_session}"
      printf '  "tmux_alive": %s,\n' "$(tmux_alive && echo true || echo false)"
    else
      printf '  "tmux_session": null,\n'
      printf '  "tmux_alive": null,\n'
    fi
    printf '  "footer_state": "%s",\n' "${footer_state}"
    printf '  "error_markers_seen": %s,\n' "${marker_count}"
    printf '  "error_marker_names": "%s",\n' "${marker_names}"
    printf '  "log_file": "%s",\n' "${log_file}"
    # Informational only. This field must never drive an alert: see the header.
    if [[ -n "${age}" ]]; then
      printf '  "log_age_seconds": %s,\n' "${age}"
    else
      printf '  "log_age_seconds": null,\n'
    fi
    printf '  "log_age_is_not_an_alert_signal": true,\n'
    printf '  "detail": "%s"\n' "${detail}"
    printf '}\n'
  } > "${tmp}"
  mv -f "${tmp}" "${status_file}"
}

check_once() {
  scan_footer
  scan_error_markers

  if process_alive; then
    local how=""
    pid_alive && how="pid ${watch_pid} alive"
    if tmux_alive; then
      how="${how:+${how}, }tmux session ${tmux_session} alive"
    fi
    write_status "running" "" "false" "${how}"
    echo "[liveness] RUNNING — ${how}; error markers: ${marker_count}"
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

if [[ "${once}" -eq 1 ]]; then
  set +e
  check_once
  rc=$?
  set -e
  exit "${rc}"
fi

echo "[liveness] watching ${watch_pid:+pid ${watch_pid}}${tmux_session:+ tmux ${tmux_session}} every ${interval}s"
echo "[liveness] status file: ${status_file}"
while true; do
  set +e
  check_once
  rc=$?
  set -e
  if [[ "${rc}" -ne "${EXIT_ALIVE}" ]]; then
    exit "${rc}"
  fi
  sleep "${interval}"
done
