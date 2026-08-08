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
# Blocker 3 (an approved ceiling that was not mechanically enforced):
#   Every paid generation command runs under `timeout`, bounded by a REQUIRED
#   --max-seconds, so a hang or runaway generation cannot run unbounded.
#
#   This script deliberately knows nothing about money. A dollar ceiling is a
#   per-run approval, not a property of reusable source: converting an approved
#   ceiling and a live pod rate into a wall-clock limit is the operator's job,
#   and the result is recorded in run evidence. Baking rates or caps in here is
#   how a superseded cap once stayed mechanically enforced after a smaller one
#   had been approved: the source kept enforcing the number nobody had agreed
#   to any more, and did it silently.
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
readonly EXIT_GENERATION_FAILED=68 # bfcl_simple.py failed or hit the wall-clock timeout

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
# category=simple_python) sharing one wall-clock budget. Dividing the ceiling
# across both is what makes --max-seconds bound the whole probe
# on *total* spend for the probe, not a cap that could be paid twice over.
readonly NUM_PAID_COMMANDS=2

usage() {
  cat <<'EOF'
Usage: launch_probe.sh --commit <40-char-sha> --max-seconds <int> \
       --out-root <dir> [--dry-run]

Required:
  --commit <sha>          Full 40-character hex commit SHA to detach-checkout.

  --out-root <dir>        Root directory under which the two probe categories'
                          --out-dir subdirectories are written.

  --max-seconds <int>     Total wall-clock ceiling in seconds for the whole
                          probe, split evenly across the two paid commands.
                          Derive it OUTSIDE this script from the approved
                          ceiling and the pod's live rate: this script knows
                          neither, deliberately. A per-run approval is not a
                          property of the source. Must be a positive integer,
                          at most MAX_SECONDS_HARD_LIMIT (a mistyped-argument
                          backstop, not a budget).

Optional:
  --dry-run               Print every command that would run, in order, and
                          exit 0 without touching git, the network, or
                          spawning generation.

This script enforces wall-clock only. The monetary ceiling is a per-run
approval recorded outside the source, and the provider-side auto-termination
is the independent external hard stop that survives this process being killed.
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
commit=""
out_root=""
max_seconds_override=""
dry_run=0

# require_value aborts *before* `shift`ing past the end of $@ or silently
# swallowing the next flag as a value (e.g. `--commit --max-seconds` should be
# reported as a missing --commit value, not consume --max-seconds as the SHA).
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
[[ -z "${max_seconds_override}" ]] && missing_flags+=("--max-seconds")
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

# --max-seconds must be a positive integer: it feeds `timeout` directly, and a
# zero/negative/garbage value would disable the only bound this script enforces.
# ---------------------------------------------------------------------------
if ! [[ "${max_seconds_override}" =~ ^[0-9]+$ ]] || [[ "${max_seconds_override}" -le 0 ]]; then
  echo "ERROR: --max-seconds must be a positive integer, got: '${max_seconds_override}'" >&2
  exit "${EXIT_USAGE}"
fi

# ---------------------------------------------------------------------------
# Blocker 3: bound every paid command by wall-clock, and print the bound BEFORE
# anything that can spend runs. The bound arrives as --max-seconds; this script
# does not compute it from money, because a dollar ceiling is a per-run approval
# rather than a property of the source. MAX_SECONDS_HARD_LIMIT is a sanity
# backstop against a mistyped argument, not a budget.
# ---------------------------------------------------------------------------
readonly MAX_SECONDS_HARD_LIMIT=7200

if [[ "${max_seconds_override}" -gt "${MAX_SECONDS_HARD_LIMIT}" ]]; then
  echo "ERROR: --max-seconds ${max_seconds_override} exceeds the ${MAX_SECONDS_HARD_LIMIT}s" >&2
  echo "       sanity limit. That is longer than any approved probe, and is far" >&2
  echo "       more likely a mistyped argument than an intention." >&2
  exit "${EXIT_USAGE}"
fi

total_max_seconds="${max_seconds_override}"
echo "WALL-CLOCK BUDGET: ${total_max_seconds}s total across ${NUM_PAID_COMMANDS} paid commands"
echo "NOTE: this script enforces wall-clock only. The monetary ceiling and the"
echo "      provider-side auto-termination are enforced outside it."
per_command_max_seconds=$(( total_max_seconds / NUM_PAID_COMMANDS ))

if [[ "${per_command_max_seconds}" -le 0 ]]; then
  echo "ERROR: derived per-command wall-clock budget is ${per_command_max_seconds}s (<=0)." >&2
  echo "       total_max_seconds=${total_max_seconds}" >&2
  echo "       Pass a larger --max-seconds." >&2
  exit "${EXIT_USAGE}"
fi

echo "====================================================================="
echo "BUDGET: wall-clock only; paid_commands=${NUM_PAID_COMMANDS}"
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
    echo "       This is the wall-clock bound doing its job, not a crash. Aborting remaining" >&2
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
# ---------------------------------------------------------------------------
# EXIT trap. The stop-the-pod reminder and the artifact inventory must appear on
# EVERY exit path, not just the happy one. The previous version printed them
# only after both generations succeeded — so a failed or timed-out run, which is
# exactly when a human is most likely to walk away from a still-billing pod,
# printed nothing. A pod left running is the one cost the in-pod wall-clock
# ceiling cannot bound.
# ---------------------------------------------------------------------------
completed_all_steps=0
started_epoch="$(date -u +%s)"

on_exit() {
  local status=$?
  local elapsed=$(( $(date -u +%s) - started_epoch ))

  echo
  echo "====================================================================="
  if [[ "${dry_run}" -eq 1 ]]; then
    # Still print the stop procedure in a dry run: it is part of the plan a
    # reviewer is being asked to approve, and hiding it would mean the most
    # cost-critical step never appears in the reviewable output.
    echo "DRY RUN — the steps below are what a real run would print on exit."
  elif [[ "${completed_all_steps}" -eq 1 && "${status}" -eq 0 ]]; then
    echo "RUN COMPLETE — elapsed ${elapsed}s"
  else
    echo "RUN DID NOT COMPLETE — exit ${status}, elapsed ${elapsed}s"
    echo "Partial evidence is preserved; it is not discarded."
  fi
  echo
  echo "STOP THE POD NOW, then CONFIRM IN THE CONSOLE THAT BILLING STOPPED."
  echo "A process that has been killed cannot"
  echo "stop its own billing — only the provider-side control can."
  echo
  echo "Record into the run evidence: actual elapsed ${elapsed}s, the actual"
  echo "hourly rate, the actual charge, and billing-stopped confirmation."
  echo
  echo "Persist these before terminating (partial files count as evidence):"
  for d in "${out_root}/study2_probe_multiple" "${out_root}/study2_probe_simple_python"; do
    for f in generations.jsonl report.md run_manifest.json; do
      if [[ "${dry_run}" -eq 1 ]]; then
        echo "  ${d}/${f}"
      elif [[ -s "${d}/${f}" ]]; then
        echo "  [present] ${d}/${f}"
      else
        echo "  [MISSING] ${d}/${f}"
      fi
    done
  done
  echo "  plus: eval/out/pip_freeze.txt eval/out/gpu.txt eval/out/image_tag.txt"
  echo "  plus: this tmux session's stdout/stderr log"
  echo "====================================================================="
  return "${status}"
}
trap on_exit EXIT

# Blocker 3, preflight half. The wall-clock bound is enforced by `timeout`, so
# a missing `timeout` binary means --max-seconds is unenforceable and the only
# remaining stop is the provider's.
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
    echo "                  --max-seconds bound is enforced by wall-clock timeout." >&2
  else
    echo "ERROR: neither 'timeout' nor 'gtimeout' is on PATH." >&2
    echo "       --max-seconds is enforced by a wall-clock timeout; without it" >&2
    echo "       the only remaining stop is the provider deadline, so this" >&2
    echo "       refuses to run rather than run unbounded." >&2
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

completed_all_steps=1
exit "${EXIT_OK}"
