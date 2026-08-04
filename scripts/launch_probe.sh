#!/usr/bin/env bash
#
# launch_probe.sh — launch the study-2 on-policy DPO probe (BFCL v4:
# "multiple" and "simple_python") on a fresh, ephemeral GPU pod.
#
# This script exists because a reviewer rejected the previous manual launch
# procedure. Each numbered block below closes one specific blocker. Read the
# comments before touching the control flow — they are the audit trail for
# why each guard is here, not just decoration.
#
# Blocker 1 (no upstream branch -> `git pull` cannot work on a fresh pod):
#   We never pull. We `git checkout --detach <FULL_SHA>` at a commit passed
#   explicitly on the command line, then ASSERT `git rev-parse HEAD` equals
#   that exact SHA and ASSERT the working tree is clean. If either assertion
#   fails we abort before doing anything that costs money.
#
# Blocker 2 (`eval/bfcl_data/` is gitignored -> absent on a fresh clone):
#   We ACQUIRE the pinned files via `eval/fetch_pinned_bfcl.py` (network
#   fetch + checksum-verified write) first, and only after that do we ever
#   pass `--verify-only`. Verification then runs again immediately before
#   *each* paid generation command, so a corrupted/tampered cache can never
#   silently ride into a paid run — any verify failure aborts before spend.
#
# Blocker 3 ($2.50 hard cap approved but not mechanically enforced):
#   We derive a wall-clock ceiling from --usd-cap and --usd-per-hour and run
#   every paid generation command under `timeout`, so a hang or runaway
#   generation cannot silently blow past the approved budget. The derived
#   budget is printed clearly before any spend-capable step runs.
#
# Blocker 4 (must be fail-closed):
#   `set -euo pipefail` plus explicit, distinct exit codes per failure class
#   (see EXIT_* below) plus every side-effecting step aborting the whole
#   script on non-zero exit — nothing is allowed to fail silently and let a
#   later, more expensive step run anyway.
#
# `--dry-run` prints every command this script would run, in order, and
# exits 0 without touching git, the network, or spawning generation. That is
# what makes this script reviewable and testable without a pod or a GPU.

set -euo pipefail
IFS=$'\n\t'

# ---------------------------------------------------------------------------
# Explicit exit codes (Blocker 4). One code per failure class so a launch
# log tells you *what kind* of thing aborted the run without needing to
# grep the full log.
# ---------------------------------------------------------------------------
readonly EXIT_OK=0
readonly EXIT_USAGE=64          # missing/malformed CLI argument
readonly EXIT_GIT_UNCLEAN=65    # checkout landed on the wrong SHA, or tree dirty
readonly EXIT_ACQUIRE_FAILED=66 # fetch_pinned_bfcl.py (acquire) failed
readonly EXIT_VERIFY_FAILED=67  # fetch_pinned_bfcl.py --verify-only failed
readonly EXIT_GENERATION_FAILED=68 # bfcl_simple.py failed or hit the budget timeout

# This script always operates on the repo it lives in, resolved from its own
# path — not the caller's $PWD — so it behaves the same no matter where it
# is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT

# Always the repo's own venv, never whatever `python` happens to resolve to
# on $PATH — a fresh pod's system python is not this project's environment,
# and silently falling back to it would defeat the point of pinning a commit.
readonly PYTHON="${REPO_ROOT}/.venv/bin/python"

# The probe runs two paid generation commands (category=multiple and
# category=simple_python) sharing one approved dollar budget. Dividing the
# derived wall-clock ceiling across both is what makes --usd-cap a true cap
# on *total* spend for the probe, not a cap that could be paid twice over.
readonly NUM_PAID_COMMANDS=2

usage() {
  cat <<'EOF'
Usage: launch_probe.sh --commit <40-char-sha> --usd-cap <float> \
       --usd-per-hour <float> --out-root <dir> \
       [--max-seconds <int>] [--dry-run]

Required:
  --commit <sha>        Full 40-character hex commit SHA to detach-checkout.
  --usd-cap <float>      Approved dollar hard cap for this probe launch.
  --usd-per-hour <float> Pod's hourly rate in USD; used to derive the
                          wall-clock kill budget from --usd-cap.
  --out-root <dir>       Root directory under which the two probe
                          categories' --out-dir subdirectories are written.

Optional:
  --max-seconds <int>    Override the TOTAL derived wall-clock ceiling
                          (seconds) instead of computing it from
                          --usd-cap / --usd-per-hour. Still split evenly
                          across the two paid generation commands.
  --dry-run               Print every command that would run, in order, and
                          exit 0 without touching git, the network, or
                          spawning generation.
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
commit=""
usd_cap=""
usd_per_hour=""
out_root=""
max_seconds_override=""
dry_run=0

# require_value aborts *before* `shift`ing past the end of $@ or silently
# swallowing the next flag as a value (e.g. `--commit --usd-cap` should be
# reported as a missing --commit value, not consume --usd-cap as the SHA).
require_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "ERROR: missing value for required flag: ${flag}" >&2
    usage >&2
    exit "${EXIT_USAGE}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit)
      require_value "--commit" "${2:-}"
      commit="$2"
      shift 2
      ;;
    --usd-cap)
      require_value "--usd-cap" "${2:-}"
      usd_cap="$2"
      shift 2
      ;;
    --usd-per-hour)
      require_value "--usd-per-hour" "${2:-}"
      usd_per_hour="$2"
      shift 2
      ;;
    --out-root)
      require_value "--out-root" "${2:-}"
      out_root="$2"
      shift 2
      ;;
    --max-seconds)
      require_value "--max-seconds" "${2:-}"
      max_seconds_override="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit "${EXIT_OK}"
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit "${EXIT_USAGE}"
      ;;
  esac
done

# Collect *all* missing required flags in one message rather than failing on
# the first — a reviewer re-running this after a rejection should not have
# to hit missing-flag errors one at a time.
missing_flags=()
[[ -z "${commit}" ]] && missing_flags+=("--commit")
[[ -z "${usd_cap}" ]] && missing_flags+=("--usd-cap")
[[ -z "${usd_per_hour}" ]] && missing_flags+=("--usd-per-hour")
[[ -z "${out_root}" ]] && missing_flags+=("--out-root")
if [[ ${#missing_flags[@]} -gt 0 ]]; then
  echo "ERROR: missing required flag(s): ${missing_flags[*]}" >&2
  usage >&2
  exit "${EXIT_USAGE}"
fi

# Blocker 1 depends on --commit being an *exact*, unambiguous, full SHA —
# never a short SHA, branch name, or tag that could resolve to something
# different tomorrow. Reject anything that is not exactly 40 hex chars.
if ! [[ "${commit}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "ERROR: --commit must be a full 40-character hex SHA, got: '${commit}'" >&2
  exit "${EXIT_USAGE}"
fi

# --usd-cap / --usd-per-hour must be positive numbers — they feed directly
# into the wall-clock budget derivation below, and a zero/negative/garbage
# value there would either divide by zero or silently grant an unbounded
# (or negative) run.
if ! [[ "${usd_cap}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v v="${usd_cap}" 'BEGIN{exit !(v>0)}'; then
  echo "ERROR: --usd-cap must be a positive number, got: '${usd_cap}'" >&2
  exit "${EXIT_USAGE}"
fi
if ! [[ "${usd_per_hour}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v v="${usd_per_hour}" 'BEGIN{exit !(v>0)}'; then
  echo "ERROR: --usd-per-hour must be a positive number, got: '${usd_per_hour}'" >&2
  exit "${EXIT_USAGE}"
fi
if [[ -n "${max_seconds_override}" ]]; then
  if ! [[ "${max_seconds_override}" =~ ^[0-9]+$ ]] || [[ "${max_seconds_override}" -le 0 ]]; then
    echo "ERROR: --max-seconds must be a positive integer, got: '${max_seconds_override}'" >&2
    exit "${EXIT_USAGE}"
  fi
fi

# ---------------------------------------------------------------------------
# Blocker 3: derive and print the wall-clock budget BEFORE anything that can
# spend money runs. hours_affordable = usd_cap / usd_per_hour; floor (not
# round) the resulting seconds so we never grant more wall-clock than the
# cap actually covers, then split evenly across the two paid commands so
# --usd-cap bounds *total* probe spend, not spend-per-category.
# ---------------------------------------------------------------------------
if [[ -n "${max_seconds_override}" ]]; then
  total_max_seconds="${max_seconds_override}"
else
  total_max_seconds=$(awk -v cap="${usd_cap}" -v rate="${usd_per_hour}" \
    'BEGIN { printf "%d", (cap / rate) * 3600 }')
fi
per_command_max_seconds=$(( total_max_seconds / NUM_PAID_COMMANDS ))

if [[ "${per_command_max_seconds}" -le 0 ]]; then
  echo "ERROR: derived per-command wall-clock budget is ${per_command_max_seconds}s (<=0)." >&2
  echo "       usd-cap=${usd_cap} usd-per-hour=${usd_per_hour} total_max_seconds=${total_max_seconds}" >&2
  echo "       Raise --usd-cap, lower --usd-per-hour, or pass a larger --max-seconds." >&2
  exit "${EXIT_USAGE}"
fi

echo "====================================================================="
echo "BUDGET: usd-cap=\$${usd_cap} usd-per-hour=\$${usd_per_hour}/hr paid_commands=${NUM_PAID_COMMANDS}"
echo "BUDGET: total_max_seconds=${total_max_seconds} per_command_max_seconds=${per_command_max_seconds}"
echo "        ($(awk -v s="${total_max_seconds}" 'BEGIN{printf "%.3f", s/3600}') hours total wall-clock ceiling)"
echo "====================================================================="

# ---------------------------------------------------------------------------
# announce prints the exact command about to run (shell-quoted, so the
# printed line is copy-pasteable). It is called unconditionally — in
# --dry-run mode these announce lines ARE the entire output; in a real run
# they double as an audit log of what actually executed.
# ---------------------------------------------------------------------------
announce() {
  local formatted
  formatted=$(printf ' %q' "$@")
  printf '+%s\n' "${formatted}"
}

# run_checked announces a command, and — unless --dry-run is set — executes
# it and aborts the whole script with a specific exit code on any non-zero
# status. Centralizing this is what makes --dry-run a true simulation: every
# side-effecting call in this script funnels through here or run_generation
# below, so nothing can execute for real while --dry-run is set.
run_checked() {
  local label="$1" code="$2"
  shift 2
  announce "$@"
  if [[ "${dry_run}" -eq 1 ]]; then
    return 0
  fi
  set +e
  "$@"
  local status=$?
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "ERROR: ${label} failed (exit ${status}) — aborting before any further spend" >&2
    exit "${code}"
  fi
}

# ---------------------------------------------------------------------------
# Blocker 1: detached checkout at the pinned commit, then assert HEAD and
# tree state. This is the ONLY step that touches git ref state.
# ---------------------------------------------------------------------------
step_git_checkout() {
  local checkout_cmd=(git -C "${REPO_ROOT}" checkout --detach "${commit}")
  local head_cmd=(git -C "${REPO_ROOT}" rev-parse HEAD)
  local status_cmd=(git -C "${REPO_ROOT}" status --porcelain)

  announce "${checkout_cmd[@]}"
  announce "${head_cmd[@]}"
  echo "    # asserted to print exactly: ${commit}"
  announce "${status_cmd[@]}"
  echo "    # asserted to print nothing (clean tree)"

  if [[ "${dry_run}" -eq 1 ]]; then
    return 0
  fi

  set +e
  "${checkout_cmd[@]}"
  local checkout_status=$?
  set -e
  if [[ "${checkout_status}" -ne 0 ]]; then
    echo "ERROR: git checkout --detach ${commit} failed (exit ${checkout_status})" >&2
    exit "${EXIT_GIT_UNCLEAN}"
  fi

  local actual_head
  actual_head="$("${head_cmd[@]}")"
  if [[ "${actual_head}" != "${commit}" ]]; then
    echo "ERROR: HEAD is ${actual_head} after checkout, expected ${commit}." >&2
    echo "       Refusing to run a paid probe against an unpinned tree." >&2
    exit "${EXIT_GIT_UNCLEAN}"
  fi

  local dirty
  dirty="$("${status_cmd[@]}")"
  if [[ -n "${dirty}" ]]; then
    echo "ERROR: working tree is dirty after checkout — refusing to run a paid probe" >&2
    echo "       against a non-reproducible tree. git status --porcelain:" >&2
    echo "${dirty}" >&2
    exit "${EXIT_GIT_UNCLEAN}"
  fi

  echo "OK: HEAD confirmed at ${commit}, working tree clean."
}

# ---------------------------------------------------------------------------
# Blocker 2: acquire pinned BFCL fixtures once, then verify (checksum
# against the frozen manifest) immediately before each paid generation call.
# ---------------------------------------------------------------------------
acquire_cmd=("${PYTHON}" "${REPO_ROOT}/eval/fetch_pinned_bfcl.py" --destination-root "${REPO_ROOT}")
verify_cmd=("${PYTHON}" "${REPO_ROOT}/eval/fetch_pinned_bfcl.py" --destination-root "${REPO_ROOT}" --verify-only)

# The two paid commands this script wraps. Flags are copied verbatim from
# the approved probe spec — do not add, remove, or reorder flags here.
gen_common_args=(
  --include-base
  --sft-only
  --sft-adapter "centuriandip/llama-3.1-8b-tools-sft"
  --sft-adapter-subfolder "adapter"
  --sft-adapter-revision "b6f4da479f8c6fc044ee8b802a92f47780f970c5"
  --base-revision "0e9e39f249a16976918f6564b8830bc894c89659"
)
gen_multiple_cmd=(
  "${PYTHON}" "${REPO_ROOT}/eval/bfcl_simple.py"
  --category multiple
  "${gen_common_args[@]}"
  --out-dir "${out_root}/study2_probe_multiple"
)
gen_simple_python_cmd=(
  "${PYTHON}" "${REPO_ROOT}/eval/bfcl_simple.py"
  --category simple_python
  "${gen_common_args[@]}"
  --out-dir "${out_root}/study2_probe_simple_python"
)

# run_generation wraps a paid command in `timeout` so the derived wall-clock
# budget (Blocker 3) is mechanically enforced rather than just documented.
# --kill-after guarantees a SIGKILL follows if the process ignores SIGTERM
# (e.g. mid CUDA-context teardown) — a timeout that doesn't actually stop
# the meter is not a cap.
run_generation() {
  local label="$1"
  shift
  announce "${timeout_bin}" --kill-after=30 "${per_command_max_seconds}" "$@"
  if [[ "${dry_run}" -eq 1 ]]; then
    return 0
  fi

  if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: expected venv python at ${PYTHON} — set up the pod's .venv first." >&2
    exit "${EXIT_USAGE}"
  fi

  echo "---- launching paid generation: ${label} (budget ${per_command_max_seconds}s) ----"
  set +e
  "${timeout_bin}" --kill-after=30 "${per_command_max_seconds}" "$@"
  local status=$?
  set -e
  if [[ "${status}" -eq 124 ]]; then
    echo "ERROR: ${label} hit the ${per_command_max_seconds}s wall-clock budget and was killed." >&2
    echo "       This is the \$${usd_cap} hard cap doing its job, not a crash. Aborting remaining" >&2
    echo "       steps rather than spending further." >&2
    exit "${EXIT_GENERATION_FAILED}"
  elif [[ "${status}" -ne 0 ]]; then
    echo "ERROR: ${label} exited with status ${status}" >&2
    exit "${EXIT_GENERATION_FAILED}"
  fi
}

# ---------------------------------------------------------------------------
# Orchestration — order matters and is exactly what --dry-run prints:
#   1. detached checkout + HEAD/clean-tree assertions      (Blocker 1)
#   2. acquire pinned BFCL fixtures                          (Blocker 2)
#   3. verify fixtures                                       (Blocker 2)
#   4. paid generation: category=multiple                    (Blocker 3)
#   5. verify fixtures again, immediately before the 2nd spend (Blocker 2)
#   6. paid generation: category=simple_python                (Blocker 3)
# Any failure at any step aborts every step after it (Blocker 4).
# ---------------------------------------------------------------------------
# Blocker 3, preflight half. The wall-clock cap is enforced by `timeout`, so a
# missing `timeout` binary means the approved $2.50 ceiling is unenforceable.
# This is checked HERE, before the detached checkout and before anything is
# fetched, because discovering it later would leave the repo on a detached HEAD
# and a pod billing for a download that can never be used. macOS ships coreutils
# as `gtimeout`; the Linux pod images have `timeout`. Accept either, fail if
# neither, and never fall back to running uncapped.
timeout_bin=""
for candidate in timeout gtimeout; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    timeout_bin="${candidate}"
    break
  fi
done
if [[ -z "${timeout_bin}" ]]; then
  if [[ "${dry_run}" -eq 1 ]]; then
    # A dry run executes nothing, and its whole purpose is to be reviewable on
    # any machine — including a reviewer's laptop. Warn loudly but continue, so
    # the printed plan stays inspectable off-pod.
    timeout_bin="timeout"
    echo "PREFLIGHT WARNING: neither 'timeout' nor 'gtimeout' found on this host." >&2
    echo "                  A real run here would REFUSE to start, because the" >&2
    echo "                  approved spend cap is enforced by wall-clock timeout." >&2
  else
    echo "ERROR: neither 'timeout' nor 'gtimeout' is on PATH." >&2
    echo "       The approved spend cap is enforced by a wall-clock timeout;" >&2
    echo "       without it the cap cannot be enforced, so this refuses to run" >&2
    echo "       rather than run uncapped." >&2
    echo "       Debian/Ubuntu pods: apt-get install coreutils." >&2
    echo "       macOS: brew install coreutils (provides gtimeout)." >&2
    exit "${EXIT_USAGE}"
  fi
fi
readonly timeout_bin
echo "PREFLIGHT: wall-clock enforcement via '${timeout_bin}'"

echo
echo "Planned steps (in order):"
step_git_checkout

echo
run_checked "acquire pinned BFCL fixtures" "${EXIT_ACQUIRE_FAILED}" "${acquire_cmd[@]}"

echo
run_checked "verify pinned BFCL fixtures (pre-flight: multiple)" "${EXIT_VERIFY_FAILED}" "${verify_cmd[@]}"

echo
run_generation "multiple" "${gen_multiple_cmd[@]}"

echo
run_checked "verify pinned BFCL fixtures (pre-flight: simple_python)" "${EXIT_VERIFY_FAILED}" "${verify_cmd[@]}"

echo
run_generation "simple_python" "${gen_simple_python_cmd[@]}"

if [[ "${dry_run}" -eq 1 ]]; then
  echo
  echo "DRY RUN: no git state changed, nothing fetched, nothing generated."
fi

cat <<EOF

=====================================================================
STOP THE POD NOW — every minute past this point burns budget beyond
the approved \$${usd_cap} cap for no additional evidence.

Persist these artifacts before terminating the pod:
  ${out_root}/study2_probe_multiple/generations.jsonl
  ${out_root}/study2_probe_multiple/report.md
  ${out_root}/study2_probe_simple_python/generations.jsonl
  ${out_root}/study2_probe_simple_python/report.md
=====================================================================
EOF

exit "${EXIT_OK}"
