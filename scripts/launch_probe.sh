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
#   --deadline-epoch, so a hang or runaway generation cannot run unbounded.
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
readonly EXIT_SMOKE_GATE_FAILED=69 # isolation ladder did not come back green
# A commit that simply does not carry one of this launcher's entry points is not
# an "unclean" tree — nothing is dirty and HEAD is exactly where it was asked to
# be. It is a launcher/commit incompatibility, and it wants its own class so the
# log does not send the operator looking for a checkout problem that isn't there.
readonly EXIT_LAUNCH_INCOMPATIBLE=70
# 71: the invocation's evidence directory or receipt could not be written. A
# success whose only record is stdout is not a success -- stdout does not
# survive the pod. 72 is taken by probe_liveness.sh (DIED HARD).
readonly EXIT_EVIDENCE_FAILED=71

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
# category=simple_python) under ONE shared deadline. Used for reporting only —
# the bound is the deadline, not a per-command quotient, because the two
# commands carry very different workloads (400 vs 800 generations).
readonly NUM_PAID_COMMANDS=2

usage() {
  cat <<'EOF'
Usage: launch_probe.sh --commit <40-char-sha> \
       --provider-deadline-epoch <int> --deadline-epoch <int> \
       --out-root <dir> [--dry-run]

Required:
  --commit <sha>          Full 40-character hex commit SHA to detach-checkout.

  --out-root <dir>        Root directory under which the two probe categories'
                          --out-dir subdirectories are written.

  --deadline-epoch <int>  Shared in-process deadline as Unix epoch seconds.
                          Derive it OUTSIDE this script by subtracting the
                          shutdown reserve from the provider deadline. Both
                          paid commands share this exact absolute deadline;
                          it never resets or shifts if launch is delayed.

  --provider-deadline-epoch <int>
                          Provider auto-termination deadline as Unix epoch
                          seconds. The script refuses to run unless its shared
                          deadline is strictly earlier, mechanically nesting
                          the in-process bound inside the external hard stop.

Optional:
  --stop-after-ladder     Run the mandatory ladder gate and stop cleanly after
                          it, without invoking either generation command. Exits
                          0 with outcome=ladder_only_green and prints the
                          remaining runway so the operator can decide whether a
                          full invocation still fits before the deadline. This
                          is a scope selector, NOT a way to skip the gate: the
                          gate always runs, and every later invocation reruns it.

  --invocation-id <str>   Label for this invocation's evidence directory under
                          <out-root>/invocations/. Defaults to a UTC timestamp.
                          Pass the same value the caller used for its log
                          redirect so one directory holds the whole invocation.

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
deadline_epoch_input=""
provider_deadline_epoch=""
dry_run=0
stop_after_ladder=0
invocation_id=""
# Named outcomes, so a reader never has to remember which integers are good.
# Exit status stays 0 for BOTH successes: probe_liveness.sh derives
# footer_state purely from the integer (0 -> complete, anything else ->
# failed), so a deliberate ladder-only stop exiting non-zero would be
# classified as a failure and the runbook would refuse to collect its
# evidence. The outcome string is what distinguishes them.
readonly OUTCOME_FULL="full_probe_complete"
readonly OUTCOME_LADDER_ONLY="ladder_only_green"
readonly OUTCOME_FAILED="failed"
probe_outcome="${OUTCOME_FAILED}"

# require_value aborts *before* `shift`ing past the end of $@ or silently
# swallowing the next flag as a value (e.g. `--commit --deadline-epoch` should
# be reported as a missing --commit value, not consume the next flag as the SHA).
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
    --deadline-epoch)
      require_value "--deadline-epoch" "${2:-}"
      deadline_epoch_input="$2"
      shift 2
      ;;
    --provider-deadline-epoch)
      require_value "--provider-deadline-epoch" "${2:-}"
      provider_deadline_epoch="$2"
      shift 2
      ;;
    --stop-after-ladder)
      stop_after_ladder=1
      shift
      ;;
    --invocation-id)
      require_value "--invocation-id" "${2:-}"
      invocation_id="$2"
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
[[ -z "${deadline_epoch_input}" ]] && missing_flags+=("--deadline-epoch")
[[ -z "${provider_deadline_epoch}" ]] && missing_flags+=("--provider-deadline-epoch")
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

# Both deadlines must be positive integer epochs. Invalid values would disable
# or invert the only bounds this script enforces.
# ---------------------------------------------------------------------------
if ! [[ "${deadline_epoch_input}" =~ ^[0-9]+$ ]] || [[ "${deadline_epoch_input}" -le 0 ]]; then
  echo "ERROR: --deadline-epoch must be a positive integer, got: '${deadline_epoch_input}'" >&2
  exit "${EXIT_USAGE}"
fi
if ! [[ "${provider_deadline_epoch}" =~ ^[0-9]+$ ]] || [[ "${provider_deadline_epoch}" -le 0 ]]; then
  echo "ERROR: --provider-deadline-epoch must be a positive integer, got: '${provider_deadline_epoch}'" >&2
  exit "${EXIT_USAGE}"
fi

# ---------------------------------------------------------------------------
# Blocker 3: bound the whole paid sequence by ONE absolute deadline, stamped
# once here and never reset.
#
# The two absolute deadlines arrive already derived. This script does not
# compute them from money: a monetary ceiling is a per-run approval, not a
# property of reusable source. Nor does it carry a sanity ceiling of its own.
# Absolute epochs are used instead of a relative duration so pausing between
# derivation and launch cannot silently move the bound later.
#
# A SHARED DEADLINE, NOT A PER-COMMAND BUDGET. An earlier version split
# the allowed duration evenly across the two paid commands, which is wrong for this
# workload: category=multiple is 200 prompts x 2 candidates = 400 generations,
# category=simple_python is 400 x 2 = 800. An even split hands the command with
# twice the work the same allowance, so the probe would reliably be killed
# mid-simple_python having already paid for it. Each command instead gets
# whatever is left of the shared deadline, so slack from a fast first command
# flows to the second and the sequence as a whole is what is bounded.
# ---------------------------------------------------------------------------
derivation_epoch=$(date +%s)
deadline_epoch="${deadline_epoch_input}"
total_max_seconds=$(( deadline_epoch - derivation_epoch ))
if [[ "${total_max_seconds}" -le 0 ]]; then
  echo "ERROR: --deadline-epoch ${deadline_epoch} has already passed" >&2
  echo "       (current epoch ${derivation_epoch}); refusing to launch." >&2
  exit "${EXIT_USAGE}"
fi
if [[ "${deadline_epoch}" -ge "${provider_deadline_epoch}" ]]; then
  echo "ERROR: script deadline ${deadline_epoch} is not earlier than provider deadline" >&2
  echo "       ${provider_deadline_epoch}. Re-derive --deadline-epoch by" >&2
  echo "       subtracting the shutdown reserve; refusing to launch." >&2
  exit "${EXIT_USAGE}"
fi
readonly total_max_seconds derivation_epoch deadline_epoch provider_deadline_epoch

# Seconds left before the shared deadline. Monotonically shrinking across the
# run by construction — there is no path that extends it.
remaining_seconds() {
  local now
  now=$(date +%s)
  echo $(( deadline_epoch - now ))
}

echo "====================================================================="
echo "BUDGET: wall-clock only; ${NUM_PAID_COMMANDS} paid commands share ONE deadline"
echo "BUDGET: derivation_epoch=${derivation_epoch} total_max_seconds=${total_max_seconds}"
echo "BUDGET: deadline_epoch=${deadline_epoch} provider_deadline_epoch=${provider_deadline_epoch}"
echo "        ($(awk -v s="${total_max_seconds}" 'BEGIN{printf "%.3f", s/3600}') hours from now, absolute)"
echo "NOTE: this script enforces wall-clock only. The monetary ceiling and the"
echo "      provider-side auto-termination are enforced outside it, and the"
echo "      provider deadline is the bound that survives this process dying."
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
#
# ERREXIT IS NEVER TURNED OFF. The previous form of this function, and of the
# two below it, wrapped the call in `set +e` / `set -e` to capture the exit
# status. That opens a window in which the script's central guarantee — Blocker
# 4, "nothing is allowed to fail silently and let a later, more expensive step
# run anyway" — is not in force, and the window is exactly where the risky
# command runs. It also restores `set -e` unconditionally rather than to its
# prior value. `cmd || status=$?` captures the same status with errexit on for
# the whole file, so "set -euo pipefail throughout" is true of the runtime and
# not only of line 46.
run_checked() {
  local label="$1" code="$2"
  shift 2
  announce "$@"
  if [[ "${dry_run}" -eq 1 ]]; then
    return 0
  fi
  local status=0
  "$@" || status=$?
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

  local checkout_status=0
  "${checkout_cmd[@]}" || checkout_status=$?
  if [[ "${checkout_status}" -ne 0 ]]; then
    echo "ERROR: git checkout --detach ${commit} failed (exit ${checkout_status})" >&2
    exit "${EXIT_GIT_UNCLEAN}"
  fi

  # rev-parse and status are classified failures too. Previously both were bare
  # command substitutions: if either git call itself failed, `set -e` aborted
  # with git's exit code and printed nothing, so a broken *assertion* was
  # indistinguishable in the log from the assertion having caught a real
  # problem. The two demand opposite responses — one is a sick pod, the other is
  # a wrong tree — and they must never present identically.
  local actual_head="" head_status=0
  actual_head="$("${head_cmd[@]}")" || head_status=$?
  if [[ "${head_status}" -ne 0 ]]; then
    echo "ERROR: git rev-parse HEAD failed (exit ${head_status}) after checkout." >&2
    echo "       The checked-out SHA cannot be asserted, so it is not asserted." >&2
    exit "${EXIT_GIT_UNCLEAN}"
  fi
  # Shape check before equality. An empty or truncated rev-parse result would
  # otherwise be reported as a SHA mismatch — blaming the checkout for a read
  # that returned nothing.
  if ! [[ "${actual_head}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "ERROR: git rev-parse HEAD returned '${actual_head}', not a 40-char SHA." >&2
    exit "${EXIT_GIT_UNCLEAN}"
  fi
  if [[ "${actual_head}" != "${commit}" ]]; then
    echo "ERROR: HEAD is ${actual_head} after checkout, expected ${commit}." >&2
    echo "       Refusing to run a paid probe against an unpinned tree." >&2
    exit "${EXIT_GIT_UNCLEAN}"
  fi

  local dirty="" dirty_status=0
  dirty="$("${status_cmd[@]}")" || dirty_status=$?
  if [[ "${dirty_status}" -ne 0 ]]; then
    echo "ERROR: git status --porcelain failed (exit ${dirty_status}) after checkout." >&2
    echo "       Tree cleanliness cannot be asserted, so it is not asserted." >&2
    exit "${EXIT_GIT_UNCLEAN}"
  fi
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
# The §0 smoke gate. This runs INSIDE the launcher, on the same node, in the
# same process tree and against the same weights cache as the paid generation
# that follows — not as a separate command an operator is trusted to remember.
#
# A gate that lives only in a runbook is not a gate: the failure it guards
# against is a full probe launched straight into the same CUDA fault that killed
# the last one, and "the operator will run the ladder first" is precisely the
# assumption that fails under time pressure on a billing pod. Placing it here
# also means a green result is same-run, same-GPU evidence rather than a receipt
# from some earlier session on some other node.
# Every invocation owns a directory. The same pod may run the ladder, pause for
# the operator to read it, then run again for generation -- and the second run
# must not overwrite the first's evidence. Two green ladders bracketing the paid
# work are a before/after health check on one node; that only works if both
# survive. A shared "latest" path would destroy exactly the comparison the pause
# exists to enable.
if [[ -z "${invocation_id}" ]]; then
  invocation_id="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if ! [[ "${invocation_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: --invocation-id must be [A-Za-z0-9._-]+, got: '${invocation_id}'" >&2
  exit "${EXIT_USAGE}"
fi
invocation_dir="${out_root}/invocations/${invocation_id}"
if [[ "${dry_run}" -eq 0 ]]; then
  mkdir -p "${invocation_dir}" \
    || { echo "ERROR: cannot create ${invocation_dir}" >&2; exit "${EXIT_EVIDENCE_FAILED}"; }
fi

ladder_cmd=(
  "${PYTHON}" "${REPO_ROOT}/eval/isolation_ladder.py"
  --out-dir "${invocation_dir}/isolation_ladder"
)

gen_multiple_cmd=(
  "${PYTHON}" "${REPO_ROOT}/eval/bfcl_simple.py"
  --category multiple
  "${gen_common_args[@]}"
  --out-dir "${invocation_dir}/study2_probe_multiple"
)
gen_simple_python_cmd=(
  "${PYTHON}" "${REPO_ROOT}/eval/bfcl_simple.py"
  --category simple_python
  "${gen_common_args[@]}"
  --out-dir "${invocation_dir}/study2_probe_simple_python"
)

# run_generation wraps a paid command in `timeout`, bounded by whatever is left
# of the shared deadline at the moment it starts — never by a fresh allowance.
# --kill-after guarantees a SIGKILL follows if the process ignores SIGTERM
# (e.g. mid CUDA-context teardown) — a timeout that doesn't actually stop the
# meter is not a bound.
#
# The remaining time is checked BEFORE spending, not after: if the deadline has
# already passed, launching would buy generation that is certain to be killed
# and is billed anyway.
# run_bounded wraps every billed model-execution command in `timeout`, bounded
# by whatever is left of the shared deadline at the moment it starts.
#
# The exit class is a parameter rather than hardcoded to EXIT_GENERATION_FAILED
# because the smoke gate is billed too and must be under the same bound, but a
# gate failure and a generation failure are different diagnoses and must not
# collapse into one code. The gate does four model loads and four generations on
# a metered pod: leaving it on the unbounded `run_checked` path meant a hung
# CUDA load could sail past the script deadline and eat the shutdown reserve
# until the provider killed the pod.
run_bounded() {
  local label="$1"
  local failure_code="$2"
  shift 2
  local budget
  budget=$(remaining_seconds)

  if [[ "${budget}" -le 0 ]]; then
    echo "ERROR: the shared wall-clock deadline passed before ${label} started" >&2
    echo "       (${budget}s remaining). Refusing to launch: this command" >&2
    echo "       would be billed and then killed. Re-derive --deadline-epoch" >&2
    echo "       from the provider deadline and shutdown reserve, then re-run." >&2
    exit "${failure_code}"
  fi

  announce "${timeout_bin}" --kill-after=30 "${budget}" "$@"
  if [[ "${dry_run}" -eq 1 ]]; then
    return 0
  fi

  if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: expected venv python at ${PYTHON} — set up the pod's .venv first." >&2
    exit "${EXIT_USAGE}"
  fi

  echo "---- launching billed command: ${label} (${budget}s left of shared deadline) ----"
  local status=0
  "${timeout_bin}" --kill-after=30 "${budget}" "$@" || status=$?
  if [[ "${status}" -eq 124 ]]; then
    echo "ERROR: ${label} exhausted the shared wall-clock deadline and was killed" >&2
    echo "       after ${budget}s. This is the bound doing its job, not a crash." >&2
    echo "       Aborting remaining steps rather than spending further." >&2
    exit "${failure_code}"
  elif [[ "${status}" -ne 0 ]]; then
    echo "ERROR: ${label} exited with status ${status}" >&2
    exit "${failure_code}"
  fi
}

# ---------------------------------------------------------------------------
# Orchestration — order matters and is exactly what --dry-run prints:
#   1. detached checkout + HEAD/clean-tree assertions      (Blocker 1)
#   2. acquire pinned BFCL fixtures                          (Blocker 2)
#   3. verify fixtures                                       (Blocker 2)
#   4. §0 isolation ladder smoke gate, wall-clock bounded    (Blocker 5)
#   5. verify fixtures again, immediately before the 1st full generation (Blocker 2)
#   6. paid generation: category=multiple                    (Blocker 3)
#   7. verify fixtures again, immediately before the 2nd spend (Blocker 2)
#   8. paid generation: category=simple_python                (Blocker 3)
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
    if [[ "${stop_after_ladder}" -eq 1 ]]; then
      echo "DRY RUN (--stop-after-ladder) — gate only; NO generation command would run."
    else
      echo "DRY RUN (full probe) — gate, then both generation commands."
    fi
    echo "DRY RUN — the steps below are what a real run would print on exit."
  elif [[ "${completed_all_steps}" -eq 1 && "${status}" -eq 0 ]]; then
    echo "RUN COMPLETE — outcome=${probe_outcome}, elapsed ${elapsed}s"
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
  for d in "${invocation_dir}/study2_probe_multiple" "${invocation_dir}/study2_probe_simple_python"; do
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
  # Pod-wide environment receipts. Written once by bootstrap, shared by every
  # invocation on this pod, and deliberately NOT per-invocation: they describe
  # the machine, not the run.
  echo "  pod-wide (bootstrap): ${out_root}/pip_freeze.txt ${out_root}/gpu.txt"
  echo "  pod-wide (bootstrap): ${out_root}/image_tag.txt ${out_root}/env_fingerprint.json"
  echo "  pod-wide (bootstrap): ${out_root}/bundle_sha256.txt ${out_root}/auto_terminate_attestation.txt"
  # This invocation's own evidence. A second invocation on the same pod writes
  # its own directory; neither overwrites the other.
  echo "  invocation ${invocation_id}: ${launcher_pid_file} (the exact pid this run published)"
  # The gate's evidence is listed even when the gate is what failed — especially
  # then. A run aborted at the ladder has no generations to persist, and its
  # entire value is in these files.
  if [[ "${dry_run}" -eq 1 ]]; then
    echo "  invocation ${invocation_id}: ${invocation_dir}/isolation_ladder/isolation_ladder.json"
    echo "  invocation ${invocation_id}: ${invocation_dir}/isolation_ladder/telemetry/"
    echo "  invocation ${invocation_id}: ${invocation_dir}/isolation_ladder/nvidia_smi_q_pre_run.txt"
  elif [[ -s "${invocation_dir}/isolation_ladder/isolation_ladder.json" ]]; then
    echo "  [present] ${invocation_dir}/isolation_ladder/isolation_ladder.json"
    echo "  [present] ${invocation_dir}/isolation_ladder/telemetry/"
  else
    echo "  [MISSING] ${invocation_dir}/isolation_ladder/isolation_ladder.json"
  fi
  if [[ "${probe_outcome}" == "${OUTCOME_LADDER_ONLY}" ]]; then
    echo "  invocation ${invocation_id}: ${invocation_dir}/ladder_only_receipt.txt"
    echo "  (ladder-only scope: no generations exist for this invocation, by design)"
  fi
  echo "  plus: this tmux session's stdout/stderr log"
  echo "====================================================================="

  # Machine-readable terminal record, carrying this shell's PID.
  #
  # scripts/probe_liveness.sh reads this line and checks the pid against the one
  # it was told to watch. Without the pid a monitor can only match on the prose
  # footer above, and a log file appended by two consecutive runs would let it
  # report the FIRST run's clean exit as the second run's outcome — a stale
  # record read as a live one, which is the failure mode this whole exercise is
  # about. Absence of this line after the process is gone is itself the signal:
  # the trap never ran, so the process did not exit in an orderly way.
  echo "PROBE_EXIT_RECORD pid=$$ exit=${status} outcome=${probe_outcome} invocation=${invocation_id:-unset} elapsed=${elapsed}s completed_all_steps=${completed_all_steps} dry_run=${dry_run}"
  return "${status}"
}
trap on_exit EXIT

# Blocker 3, preflight half. The wall-clock bound is enforced by `timeout`, so
# a missing `timeout` binary means --deadline-epoch is unenforceable and the only
# remaining stop is the provider's.
# This is checked HERE, before the detached checkout and before anything is
# fetched, because discovering it later would leave the repo on a detached HEAD
# and a pod billing for a download that can never be used. macOS ships coreutils
# as `gtimeout`; the Linux pod images have `timeout`. Accept either, fail if
# neither, and never fall back to running uncapped.
# `command -v` writes the resolved path to stdout and nothing to stderr, so the
# `2>&1` this used to carry never discarded a diagnostic. It is dropped anyway:
# leaving one instance in the file makes the pattern citable as precedent, and
# the pattern is what this audit exists to remove.
timeout_bin=""
for candidate in timeout gtimeout; do
  if command -v "${candidate}" >/dev/null; then
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
    echo "                  --deadline-epoch is enforced by wall-clock timeout." >&2
  else
    echo "ERROR: neither 'timeout' nor 'gtimeout' is on PATH." >&2
    echo "       --deadline-epoch is enforced by a wall-clock timeout; without it" >&2
    echo "       the only remaining stop is the provider deadline, so this" >&2
    echo "       refuses to run rather than run unbounded." >&2
    echo "       Debian/Ubuntu pods: apt-get install coreutils." >&2
    echo "       macOS: brew install coreutils (provides gtimeout)." >&2
    exit "${EXIT_USAGE}"
  fi
fi
readonly timeout_bin
echo "PREFLIGHT: wall-clock enforcement via '${timeout_bin}'"

# Blocker 4, existence half. Every command this script runs is `${PYTHON}` plus
# a script path, and neither was ever asserted to exist. A missing interpreter
# or entry point therefore surfaced as whatever the *first* step that used it
# happened to report: bash's "No such file or directory" (exit 127) funnelled
# through run_checked and relabelled "acquire pinned BFCL fixtures failed",
# which names the wrong thing. On a billing pod the operator then debugs the
# fetcher instead of the venv.
#
# The interpreter is checked here, before the detached checkout, for the same
# reason the timeout binary is: discovering it afterwards leaves the repo on a
# detached HEAD with nothing to run.
if [[ "${dry_run}" -eq 0 ]]; then
  if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: no executable interpreter at ${PYTHON}." >&2
    echo "       This script never falls back to whatever 'python' is on PATH —" >&2
    echo "       a fresh pod's system python is not this project's environment." >&2
    echo "       Run scripts/bootstrap_pod.sh first." >&2
    exit "${EXIT_USAGE}"
  fi
  echo "PREFLIGHT: interpreter ${PYTHON}"
else
  echo "PREFLIGHT: would assert an executable interpreter at ${PYTHON}"
fi

# assert_entrypoint checks one script path and says which step needed it.
# Called AFTER the detached checkout, because the question is whether the
# REVIEWED tree carries these files — asserting against the pre-checkout tree
# would answer a question nobody asked.
#
# WHAT THIS CHECKS DEPENDS ON THE MODE, AND IT SAYS WHICH.
#
# A real run reaches here only after step_git_checkout has asserted HEAD equals
# --commit and the tree is clean, so the working tree IS the commit's tree and a
# plain `-f` test is a statement about the commit.
#
# A dry run performs no checkout. The previous version tested the same `-f` on
# whatever the current working tree happened to be and then printed "all three
# entry points present at ${commit}" — a claim about a commit it had not looked
# at. `--commit 000…000 --dry-run` produced a confident OK for a SHA that does
# not exist. So in dry-run the commit's tree is read directly and read-only via
# `git cat-file`, and when the commit is not in this repo at all the output says
# that instead of claiming anything.
entrypoint_check_mode=""        # set by assert_entrypoint, reported in the summary
entrypoint_check_diagnostic=""  # git's own words when it could not resolve the commit

assert_entrypoint() {
  local rel="$1" purpose="$2"

  if [[ "${dry_run}" -eq 0 ]]; then
    entrypoint_check_mode="the checked-out tree at ${commit}"
    if [[ ! -f "${REPO_ROOT}/${rel}" ]]; then
      echo "ERROR: ${purpose} entry point is missing: ${rel}" >&2
      echo "       HEAD is ${commit} and the tree is clean, so this commit does" >&2
      echo "       not carry a file this launcher requires. That is a launcher/" >&2
      echo "       commit incompatibility, not a bad checkout: either the SHA" >&2
      echo "       predates the file, or this launcher is newer than the tree." >&2
      exit "${EXIT_LAUNCH_INCOMPATIBLE}"
    fi
    return 0
  fi

  # Dry run. Is the commit even present locally to be inspected?
  #
  # git's stderr is CAPTURED and then ACTUALLY PRINTED. The previous version
  # captured it and threw it away, then mapped every non-zero result onto the
  # single sentence "commit is not in this repository" — so a git that failed
  # for any other reason (a broken object database, an unreadable repo, an I/O
  # error) was reported as a clean, ordinary absence, and the one line
  # explaining what really happened was discarded. Capturing a diagnostic and
  # not showing it is the same defect as suppressing it, wearing a disguise.
  #
  # The label does not overclaim, because it CANNOT be resolved from the exit
  # code: `git cat-file -e` returns 128 both for a commit that does not exist
  # and for a repository it could not read. So this says only what is true —
  # the commit could not be resolved — and hands the operator git's own words
  # to tell the two apart.
  local err="" status=0
  err="$(git -C "${REPO_ROOT}" cat-file -e "${commit}^{commit}" 2>&1)" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    entrypoint_check_mode="NOT VERIFIED — git could not resolve commit ${commit} (exit ${status})"
    entrypoint_check_diagnostic="${err}"
    return 0
  fi

  entrypoint_check_mode="commit ${commit}, read-only via git cat-file"
  status=0
  err="$(git -C "${REPO_ROOT}" cat-file -e "${commit}:${rel}" 2>&1)" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "ERROR: ${purpose} entry point is missing from commit ${commit}: ${rel}" >&2
    echo "       Read directly from the commit's tree, so this is not a working-" >&2
    echo "       directory artefact. A real run would abort here having spent" >&2
    echo "       nothing. Launcher/commit incompatibility." >&2
    [[ -n "${err}" ]] && echo "       git said: ${err}" >&2
    exit "${EXIT_LAUNCH_INCOMPATIBLE}"
  fi
}

# Publish this shell's PID before any risky work, so the monitor is handed an
# exact number instead of guessing with `pgrep -n -f`, which cannot recover a
# launcher that has already died and can match an unrelated process.
#
# This is NOT a liveness claim and must never be read as one: a process cannot
# update a file after being SIGKILLed, which is exactly how the 2026-08-08 probe
# came to have a PID file pointing at nothing. The file is the *source of the
# number*; probe_liveness.sh still decides liveness with `kill -0` on it.
launcher_pid_file="${invocation_dir}/launcher.pid"
if [[ "${dry_run}" -eq 1 ]]; then
  echo "DRY RUN: would write this launcher's PID to ${launcher_pid_file}"
else
  mkdir -p "${out_root}"
  # Written atomically: a monitor reading a half-written pid would kill -0 a
  # truncated number, i.e. some other process entirely.
  printf '%s\n' "$$" > "${launcher_pid_file}.tmp.$$"
  mv -f "${launcher_pid_file}.tmp.$$" "${launcher_pid_file}"
  # Asserted, not assumed. The monitor is started from this file; if it is
  # absent the operator gets "PID file not found" minutes later and has no way
  # to recover the number, because the process that knew it is the one being
  # watched. A launch that cannot be monitored must not proceed to spend.
  if [[ ! -s "${launcher_pid_file}" ]]; then
    echo "ERROR: ${launcher_pid_file} is missing or empty after being written." >&2
    echo "       The run would be unmonitorable; refusing to proceed to spend." >&2
    exit "${EXIT_USAGE}"
  fi
  echo "Launcher PID $$ recorded at ${launcher_pid_file}"
fi

echo
echo "Planned steps (in order):"
step_git_checkout

# Checked in BOTH modes, deliberately. A dry run whose printed plan references a
# script the commit does not carry is not a reviewable plan — it is a plan that
# will fail on the pod, reviewed as though it would work.
assert_entrypoint "eval/fetch_pinned_bfcl.py" "fixture acquire/verify"
assert_entrypoint "eval/isolation_ladder.py"  "§0 smoke gate"
assert_entrypoint "eval/bfcl_simple.py"       "paid generation"
# The summary names what was actually inspected. It used to say "present at
# ${commit}" unconditionally, which in a dry run was a claim about a commit that
# had never been read — and was printed even for a SHA that does not exist.
if [[ "${entrypoint_check_mode}" == NOT\ VERIFIED* ]]; then
  echo "WARNING: entry points ${entrypoint_check_mode}." >&2
  # git's own message, printed rather than swallowed. It is what distinguishes
  # "that SHA does not exist here" from "this repository is broken", which the
  # exit code cannot: cat-file returns 128 for both.
  if [[ -n "${entrypoint_check_diagnostic}" ]]; then
    echo "         git said: ${entrypoint_check_diagnostic}" >&2
  fi
  echo "         The plan below is printed UNVERIFIED against that SHA. If the" >&2
  echo "         message above is anything other than an unknown object, treat" >&2
  echo "         this repository as suspect before trusting any dry run from it." >&2
  echo "         A real run cannot reach this state: it checks out first." >&2
else
  echo "OK: all three entry points present in ${entrypoint_check_mode}."
fi

echo
run_checked "acquire pinned BFCL fixtures" "${EXIT_ACQUIRE_FAILED}" "${acquire_cmd[@]}"

echo
run_checked "verify pinned BFCL fixtures (pre-flight: multiple)" "${EXIT_VERIFY_FAILED}" "${verify_cmd[@]}"

# The gate. Runs after the fixtures exist (it reads the first `multiple` prompt)
# and before the full probe. A non-zero exit aborts here, so a run that would
# have reproduced the §0 crash spends four model loads and 32 generated tokens
# instead of 1,200 generations.
#
# It goes through run_bounded, not run_checked: the ladder is itself a billed
# command that loads an 8B model four times and generates, so an unbounded gate
# could hang past the script deadline and consume the shutdown reserve.
echo
run_bounded "§0 isolation ladder (smoke gate)" "${EXIT_SMOKE_GATE_FAILED}" "${ladder_cmd[@]}"

# Verify AGAIN, immediately before the first full generation. Inserting the gate
# between the earlier verify and this generation broke the standing invariant
# that a checksum check sits immediately before *each* paid generation, with
# nothing in between — and the thing now in between loads 16 GB of weights and
# writes to the same volume. Restoring the invariant costs a checksum pass.
echo
# --stop-after-ladder ends here: the gate has run and passed, and no generation
# command has been invoked. This is a SUCCESS of the selected scope, so it exits
# 0 -- see the OUTCOME_* comment above for why a distinct non-zero code would
# make the monitor classify it as a failure and block its own evidence.
#
# What is printed is runway, not a countdown. It asserts nothing and terminates
# nothing; the operator compares the numbers and decides. A second timer that
# could stop the pod would be a mechanism that can silently fail, and the whole
# reason this pause is safe is that the provider deadline is the only hard stop.
if [[ "${stop_after_ladder}" -eq 1 ]]; then
  now_epoch="$(date -u +%s)"
  script_remaining=$(( deadline_epoch - now_epoch ))
  provider_remaining=$(( provider_deadline_epoch - now_epoch ))
  if ! now_utc="$(date -u -d "@${now_epoch}" +%Y-%m-%dT%H:%M:%SZ)" \
     || ! script_deadline_utc="$(date -u -d "@${deadline_epoch}" +%Y-%m-%dT%H:%M:%SZ)" \
     || ! provider_deadline_utc="$(date -u -d "@${provider_deadline_epoch}" +%Y-%m-%dT%H:%M:%SZ)"; then
    echo "ERROR: GNU date could not format the deadlines. The runway figures" >&2
    echo "       are the whole point of this stop; printing epochs alone would" >&2
    echo "       leave the operator to convert them by hand under time pressure." >&2
    exit "${EXIT_EVIDENCE_FAILED}"
  fi

  receipt="${invocation_dir}/ladder_only_receipt.txt"
  if [[ "${dry_run}" -eq 1 ]]; then
    echo
    echo "DRY RUN: would stop here after a green ladder, write ${receipt},"
    echo "         print the remaining runway, and exit 0 with"
    echo "         outcome=${OUTCOME_LADDER_ONLY}. No generation command runs."
    probe_outcome="${OUTCOME_LADDER_ONLY}"
    completed_all_steps=1
    exit "${EXIT_OK}"
  fi
  {
    printf '%s\n' "schema=ladder_only_receipt/v1"
    printf '%s\n' "invocation_id=${invocation_id}"
    printf '%s\n' "outcome=${OUTCOME_LADDER_ONLY}"
    printf '%s\n' "commit=${commit}"
    printf '%s\n' "ladder_green_epoch=${now_epoch}"
    printf '%s\n' "script_deadline_epoch=${deadline_epoch}"
    printf '%s\n' "provider_deadline_epoch=${provider_deadline_epoch}"
    printf '%s\n' "script_remaining_seconds=${script_remaining}"
    printf '%s\n' "provider_remaining_seconds=${provider_remaining}"
    printf '%s\n' "ladder_dir=${invocation_dir}/isolation_ladder"
  } > "${receipt}.tmp.$$" && mv -f "${receipt}.tmp.$$" "${receipt}" || {
    rm -f "${receipt}.tmp.$$"
    echo "ERROR: could not write ${receipt}." >&2
    echo "       Refusing to report a ladder-only success whose receipt does" >&2
    echo "       not exist: the runway numbers below would be the only record," >&2
    echo "       and stdout does not survive the pod." >&2
    exit "${EXIT_EVIDENCE_FAILED}"
  }

  echo
  echo "======================================================================"
  echo "LADDER-ONLY COMPLETE — gate green, no generation invoked"
  echo "======================================================================"
  echo "  invocation      : ${invocation_id}"
  echo "  evidence        : ${invocation_dir}"
  echo "  receipt         : ${receipt}"
  echo
  echo "  now             : ${now_utc} (${now_epoch})"
  echo "  script deadline : ${script_deadline_utc} (${deadline_epoch}) — ${script_remaining}s left"
  echo "  provider deadline: ${provider_deadline_utc} (${provider_deadline_epoch}) — ${provider_remaining}s left"
  echo
  echo "  A full second invocation needs, from its own start: the ladder again"
  echo "  (mandatory, never skipped), then both generation commands, then the"
  echo "  shutdown reserve. It refuses on its own if that does not fit -- this"
  echo "  script does not decide for you and does not stop the pod."
  echo
  echo "  No duration is hardcoded here. Compare the runway above against your"
  echo "  own measured ladder time from this invocation."
  echo "======================================================================"

  probe_outcome="${OUTCOME_LADDER_ONLY}"
  completed_all_steps=1
  exit "${EXIT_OK}"
fi

run_checked "verify pinned BFCL fixtures (post-gate, pre-flight: multiple)" "${EXIT_VERIFY_FAILED}" "${verify_cmd[@]}"

echo
run_bounded "paid generation: multiple" "${EXIT_GENERATION_FAILED}" "${gen_multiple_cmd[@]}"

echo
run_checked "verify pinned BFCL fixtures (pre-flight: simple_python)" "${EXIT_VERIFY_FAILED}" "${verify_cmd[@]}"

echo
run_bounded "paid generation: simple_python" "${EXIT_GENERATION_FAILED}" "${gen_simple_python_cmd[@]}"

if [[ "${dry_run}" -eq 1 ]]; then
  echo
  echo "DRY RUN: no git state changed, nothing fetched, nothing generated."
fi

completed_all_steps=1
probe_outcome="${OUTCOME_FULL}"
exit "${EXIT_OK}"
