#!/usr/bin/env bash
#
# Fresh-pod bootstrap for the study-2 qualification probe.
#
# Turns a bare RunPod instance into a launch-ready checkout of an EXACT
# reviewed commit, transferred as a hash-verified git bundle so nothing is
# published. Every check below is fail-closed: it exits non-zero rather than
# continuing in a state that would waste billed time or produce evidence we
# cannot trace.
#
# BILLING REALITY, stated up front because an earlier draft of the operator
# guide got this wrong: pod billing starts when the pod starts. Cloning,
# creating the venv, installing packages and downloading weights are all
# BILLED time and all draw on the same run lifecycle budget as generation.
# There is no free "until launch" phase. That is why --auto-terminate-set must
# be acknowledged before this script does anything else: the provider-side
# deadline has to already exist by the time the pod is running.
#
# Usage:
#   scripts/bootstrap_pod.sh \
#     --bundle /workspace/llama-tools.bundle \
#     --bundle-sha256-file /workspace/llama-tools.bundle.sha256 \
#     --commit <FULL_40_CHAR_SHA> \
#     --out-root /workspace/persist/study2 \
#     --auto-terminate-set "<ISO8601-deadline-Z>@<rate-from-console>" \
#     [--dry-run]
#
set -euo pipefail

readonly EXIT_OK=0
readonly EXIT_USAGE=64
readonly EXIT_PROVIDER_CAP=65   # provider-side termination not acknowledged
readonly EXIT_BUNDLE=66         # bundle missing or hash mismatch
readonly EXIT_GIT=67            # clone landed on the wrong SHA / dirty tree
readonly EXIT_ENV=68            # venv, versions, CUDA, HF or timeout preflight failed

bundle=""
bundle_sha_file=""
commit=""
out_root=""
auto_terminate_set=""
dry_run=0

# die takes (message, exit_code). The previous form was `echo "ERROR: $*"`, which
# expanded BOTH arguments into the message, so every classified failure printed
# its own exit code as if it were part of the sentence:
#   "ERROR: bundle not found: /workspace/x.bundle 66"
# A diagnostic that appends a stray integer to its own text is a small lie in the
# one place an operator reads under time pressure on a billing pod.
die() {
  local message="$1"
  local code="${2:-$EXIT_USAGE}"
  echo "ERROR: ${message}" >&2
  exit "${code}"
}

require_value() {
  [[ -n "${2:-}" && "${2:0:2}" != "--" ]] || die "$1 requires a value"
}

usage() {
  sed -n '3,26p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)              require_value "$1" "${2:-}"; bundle="$2"; shift 2 ;;
    --bundle-sha256-file)  require_value "$1" "${2:-}"; bundle_sha_file="$2"; shift 2 ;;
    --commit)              require_value "$1" "${2:-}"; commit="$2"; shift 2 ;;
    --out-root)            require_value "$1" "${2:-}"; out_root="$2"; shift 2 ;;
    --auto-terminate-set)  require_value "$1" "${2:-}"; auto_terminate_set="$2"; shift 2 ;;
    --dry-run)             dry_run=1; shift ;;
    -h|--help)             usage; exit "${EXIT_OK}" ;;
    *)                     die "unknown argument: $1" ;;
  esac
done

missing=()
[[ -n "${bundle}" ]]             || missing+=("--bundle")
[[ -n "${bundle_sha_file}" ]]    || missing+=("--bundle-sha256-file")
[[ -n "${commit}" ]]             || missing+=("--commit")
[[ -n "${out_root}" ]]           || missing+=("--out-root")
[[ -n "${auto_terminate_set}" ]] || missing+=("--auto-terminate-set")
[[ ${#missing[@]} -eq 0 ]] || die "missing required flags: ${missing[*]}"

[[ "${commit}" =~ ^[0-9a-f]{40}$ ]] \
  || die "--commit must be a full 40-char lowercase hex SHA, got: '${commit}'"

announce() { echo "+ $*"; }

# run_classified is the ONLY way this script executes a side-effecting command.
#
# It replaces a plain `run()` helper that leaned on `set -e`, which aborts with
# the *tool's* exit code: a failed clone exits 128, which is git's opinion about
# git and says nothing about which of this script's guarantees was violated. On
# a pod where every second is billed, the operator needs the class in the first
# line, not after reading the log. `run()` is deleted rather than kept unused —
# an unclassified helper sitting in the file is what the next edit reaches for,
# and it would reintroduce exactly the class this audit removed.
#
# Its dry-run guard was also `[[ cond ]] && return 0`, which is correct only
# because bash exempts the non-final members of an AND-list from `set -e`: the
# behaviour depended on a subtlety of errexit rather than on the code saying
# what it meant. Written as an `if`, a future edit that moves the line cannot
# silently change it.
run_classified() {
  local label="$1" code="$2"
  shift 2
  announce "$@"
  if [[ "${dry_run}" -eq 1 ]]; then
    return 0
  fi
  local status=0
  "$@" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    die "${label} failed (exit ${status})" "${code}"
  fi
}

# assert_file exists so "the step ran without erroring" is never mistaken for
# "the step produced its artifact". That distinction is not hypothetical here:
# the retained mining-pilot artifact directory has no env_fingerprint.json,
# pip_freeze.txt or gpu.txt. Nothing failed loudly at the time; the files just
# were not there, and nothing asserted that they should be.
#
# What that actually cost, stated precisely — an earlier version of this comment
# said the pilot's versions were "not traceably measurable today", which is
# false. They were recoverable from owner-pasted console output in the chat
# archive, and the comparison against the probe has since been completed:
# transformers, peft and accelerate all match, and torch does not (2.8.0 against
# the probe's 2.9.1). The cost of the missing receipts is that the evidence
# lives in a chat log instead of beside the run it describes — recoverable by
# someone who knows to look, and lost to everyone else.
assert_file() {
  local path="$1" what="$2" code="$3"
  [[ -f "${path}" ]] || die "${what} was not created at ${path}" "${code}"
  [[ -s "${path}" ]] || die "${what} at ${path} is empty" "${code}"
}

# ---------------------------------------------------------------------------
# STEP 0 — provider spend cap. FIRST, because billing is already running.
#
# A process that has been SIGKILLed cannot stop its own billing, and an owner
# approval authorises an amount; it does not enforce one. Only the provider-side
# deadline does. The approved figure lives in the run approval and the operator
# guide, never here: source that carries a number keeps enforcing it after the
# approval it encoded has been superseded. This script cannot set it (no
# account credentials, by design) and cannot verify it from inside the pod, so
# it requires an explicit acknowledgement string and records it as evidence.
# That is deliberately a human attestation, not a simulated check.
# ---------------------------------------------------------------------------
echo "====================================================================="
echo "STEP 0 — provider auto-termination"
if [[ ! "${auto_terminate_set}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z@.+ ]]; then
  echo "ERROR: --auto-terminate-set must look like" >&2
  echo "       <ISO8601-deadline-Z>@<rate>, with both taken from the console for" >&2
  echo "       THIS pod; a remembered rate is not evidence about this run." >&2
  echo "       Set the RunPod auto-terminate deadline FIRST, then record it here." >&2
  echo "       Billing is already running; there is no unbilled setup phase." >&2
  exit "${EXIT_PROVIDER_CAP}"
fi
echo "  acknowledged: ${auto_terminate_set}"
echo "  (attestation by the operator; not verifiable from inside the pod)"

# ---------------------------------------------------------------------------
# STEP 1 — image identity. An unknown image tag must FAIL, not be recorded as
# "unknown" and treated as evidence. A run whose environment cannot be named
# cannot be reproduced, and unreproducible evidence is the failure mode this
# whole procedure exists to prevent.
# ---------------------------------------------------------------------------
echo
echo "STEP 1 — image identity"
image_tag="${RUNPOD_IMAGE_NAME:-}"
if [[ -z "${image_tag}" || "${image_tag}" == "unknown" ]]; then
  if [[ "${dry_run}" -eq 1 ]]; then
    echo "  DRY RUN: RUNPOD_IMAGE_NAME unset locally; a real pod must export it."
    image_tag="DRY-RUN-PLACEHOLDER"
  else
    die "RUNPOD_IMAGE_NAME is unset or 'unknown'. Name the template explicitly \
(export RUNPOD_IMAGE_NAME=...) so the environment is reproducible." "${EXIT_ENV}"
  fi
fi
echo "  image: ${image_tag}"

# ---------------------------------------------------------------------------
# STEP 2 — persistent output root. Container disk is destroyed on stop; that
# is exactly how the study-1 pod lost outputs/sft-full. Evidence is written
# here from the very first preflight artifact, not copied at the end and
# hoped for.
# ---------------------------------------------------------------------------
echo
echo "STEP 2 — persistent evidence root"
run_classified "mkdir -p ${out_root}" "${EXIT_ENV}" mkdir -p "${out_root}"
if [[ "${dry_run}" -eq 0 ]]; then
  # The `2>/dev/null` that used to be on this touch discarded the only text that
  # distinguishes the failure modes: read-only filesystem, permission denied,
  # ENOSPC, or a path component that is not a directory. All four collapsed into
  # the same "is not writable" sentence, and the one telling detail — the volume
  # was never mounted — looked identical to a permissions problem.
  [[ -d "${out_root}" ]] \
    || die "--out-root ${out_root} does not exist after mkdir -p" "${EXIT_ENV}"
  touch "${out_root}/.write_probe" \
    || die "--out-root ${out_root} is not writable (the touch error above is the reason)" \
           "${EXIT_ENV}"
  [[ -f "${out_root}/.write_probe" ]] \
    || die "touch reported success but ${out_root}/.write_probe does not exist" "${EXIT_ENV}"
  rm -f "${out_root}/.write_probe"
fi
echo "  evidence root: ${out_root}"
echo "  NOTE: this must be a MOUNTED PERSISTENT VOLUME, not container disk."

# ---------------------------------------------------------------------------
# STEP 3 — bundle receipt verified BEFORE clone.
# The sidecar hash is produced on the owner's machine and travels separately.
# Verifying after cloning would mean trusting objects we have not yet checked.
# ---------------------------------------------------------------------------
echo
echo "STEP 3 — bundle transfer receipt"
if [[ "${dry_run}" -eq 0 ]]; then
  [[ -f "${bundle}" ]]          || die "bundle not found: ${bundle}" "${EXIT_BUNDLE}"
  [[ -f "${bundle_sha_file}" ]] || die "sidecar not found: ${bundle_sha_file}" "${EXIT_BUNDLE}"
  # The WHOLE receipt, with only its OUTER whitespace trimmed.
  #
  # Two rounds of this check were wrong in the same direction — each normalised
  # the input until the assertion could no longer fail on the case that mattered:
  #
  #   `tr -d '[:space:]' | cut -c1-64` — `cut` threw away everything past
  #   character 64, so a digest followed by anything at all was truncated to the
  #   valid prefix and accepted. The regex was validating cut's output, not the
  #   file.
  #
  #   `tr -d '[:space:]'` alone — still strips INTERNAL whitespace, so a digest
  #   split across two 32-character lines was reassembled into a valid one. The
  #   bootstrap printed "bundle sha256 verified" and advanced to the clone.
  #
  # A digest that arrives in two pieces is not a digest that arrived; whatever
  # produced or transported it did something nobody intended, and that is the
  # signal. Only the outer whitespace is trimmed now — `$(< file)` drops trailing
  # newlines, and the parameter expansions drop leading/trailing spaces and tabs.
  # Any whitespace left inside the string fails the character-class check,
  # because a newline is not in [0-9a-f].
  expected_sha="$(< "${bundle_sha_file}")"
  expected_sha="${expected_sha#"${expected_sha%%[![:space:]]*}"}"   # leading
  expected_sha="${expected_sha%"${expected_sha##*[![:space:]]}"}"   # trailing
  [[ "${expected_sha}" =~ ^[0-9a-f]{64}$ ]] \
    || die "sidecar ${bundle_sha_file} must contain exactly one 64-char lowercase hex digest and nothing else — no second digest, no filename, no line break inside it (read ${#expected_sha} chars: '${expected_sha}')" \
           "${EXIT_BUNDLE}"
  # `command -v` writes its result to stdout and nothing to stderr, so the
  # discarded stream here never carried a diagnostic. Dropping the `2>&1` costs
  # nothing and removes the pattern from the file entirely, so a future reader
  # cannot cite it as precedent for suppressing a stream that does matter.
  if command -v sha256sum >/dev/null; then
    actual_sha="$(sha256sum "${bundle}" | awk '{print $1}')"
  else
    actual_sha="$(shasum -a 256 "${bundle}" | awk '{print $1}')"
  fi
  [[ "${actual_sha}" == "${expected_sha}" ]] \
    || die "bundle hash mismatch: got ${actual_sha}, expected ${expected_sha}" "${EXIT_BUNDLE}"
  echo "  bundle sha256 verified: ${actual_sha}"
  echo "${actual_sha}" > "${out_root}/bundle_sha256.txt"
  assert_file "${out_root}/bundle_sha256.txt" "bundle hash receipt" "${EXIT_ENV}"
else
  announce sha256sum "${bundle}" "# compared against ${bundle_sha_file}"
fi

# ---------------------------------------------------------------------------
# STEP 4 — clone and detach at the exact reviewed SHA.
# ---------------------------------------------------------------------------
echo
echo "STEP 4 — clone at the reviewed commit"
run_classified "git clone from ${bundle}" "${EXIT_GIT}" git clone "${bundle}" llama-tools

if [[ "${dry_run}" -eq 0 ]]; then
  # Assert the clone produced a repository before asking that repository
  # anything. Without this, a clone that half-succeeded made the NEXT command
  # the one that failed, and its error ("not a git repository") reads as a
  # working-directory mistake rather than as the clone having failed.
  [[ -d llama-tools/.git ]] \
    || die "git clone reported success but llama-tools/.git does not exist" "${EXIT_GIT}"

  # Assert the bundle actually CONTAINS the reviewed commit, before checkout.
  # A bundle built from the wrong ref clones fine and then fails at checkout
  # with git's own "reference is not a tree" — which sounds like a corrupt repo.
  # The real fault is upstream, on the machine that built the bundle, and the
  # operator needs to be told that rather than debugging the pod.
  bundle_has_commit=0
  git -C llama-tools cat-file -e "${commit}^{commit}" || bundle_has_commit=$?
  [[ "${bundle_has_commit}" -eq 0 ]] \
    || die "the bundle does not contain commit ${commit}; it was built from the wrong ref on the owner's machine" \
           "${EXIT_GIT}"
fi

run_classified "git checkout --detach ${commit}" "${EXIT_GIT}" \
  git -C llama-tools checkout --detach "${commit}"

if [[ "${dry_run}" -eq 0 ]]; then
  # rev-parse and status are themselves classified. Left to `set -e` they abort
  # with git's exit code and no message at all, so a failure of the *assertion
  # machinery* was indistinguishable in the log from a failure of the thing it
  # asserts — the checkout landing on the wrong SHA.
  head_sha=""
  head_status=0
  head_sha="$(git -C llama-tools rev-parse HEAD)" || head_status=$?
  [[ "${head_status}" -eq 0 ]] \
    || die "git rev-parse HEAD failed (exit ${head_status}); the checked-out SHA cannot be asserted" \
           "${EXIT_GIT}"
  # Shape-check before comparing: an empty or truncated rev-parse result would
  # otherwise report as a plain SHA mismatch, blaming the checkout for a read
  # that never returned anything.
  [[ "${head_sha}" =~ ^[0-9a-f]{40}$ ]] \
    || die "git rev-parse HEAD returned '${head_sha}', not a 40-char SHA" "${EXIT_GIT}"
  [[ "${head_sha}" == "${commit}" ]] \
    || die "HEAD is ${head_sha}, expected ${commit}" "${EXIT_GIT}"

  dirty=""
  dirty_status=0
  dirty="$(git -C llama-tools status --porcelain)" || dirty_status=$?
  [[ "${dirty_status}" -eq 0 ]] \
    || die "git status --porcelain failed (exit ${dirty_status}); tree cleanliness cannot be asserted" \
           "${EXIT_GIT}"
  if [[ -n "${dirty}" ]]; then
    echo "${dirty}" >&2
    die "working tree is dirty immediately after clone (see git status above)" "${EXIT_GIT}"
  fi
  echo "  HEAD asserted: ${head_sha}"
fi

# ---------------------------------------------------------------------------
# STEP 5 — venv. Created HERE, before any repo script can assume one exists.
# .venv is gitignored, so it never travels in the bundle.
# ---------------------------------------------------------------------------
echo
echo "STEP 5 — virtualenv + exact probe dependency spec"
# --system-site-packages is load-bearing, not incidental. requirements-probe.txt
# deliberately omits torch so the template's CUDA build is used rather than
# overwritten -- but a plain `python3 -m venv` is ISOLATED, so that torch would
# not be importable and Step 6 would fail on every normal template. This flag is
# what makes "inherit the image's torch" actually true instead of merely
# intended.
# The dependency spec is asserted to exist BEFORE the venv is built. pip's own
# "could not open requirements file" arrives after the venv and a pip upgrade
# have already been paid for, and reads as a pip problem rather than as the
# checked-out tree being wrong.
if [[ "${dry_run}" -eq 0 ]]; then
  [[ -f llama-tools/requirements-probe.txt ]] \
    || die "llama-tools/requirements-probe.txt is missing at ${commit}; the checkout is not the reviewed tree" \
           "${EXIT_GIT}"
  [[ -f llama-tools/eval/environment_fingerprint.py ]] \
    || die "llama-tools/eval/environment_fingerprint.py is missing at ${commit}; locale provenance cannot be collected" \
           "${EXIT_GIT}"
fi

run_classified "python3 -m venv" "${EXIT_ENV}" \
  python3 -m venv --system-site-packages llama-tools/.venv

# `venv` can exit 0 having produced an unusable environment (ensurepip failure
# on a stripped image is the common one). Assert the interpreter and pip are
# actually there and executable rather than inferring it from the exit code.
if [[ "${dry_run}" -eq 0 ]]; then
  [[ -x llama-tools/.venv/bin/python ]] \
    || die "venv reported success but llama-tools/.venv/bin/python is missing or not executable" \
           "${EXIT_ENV}"
  [[ -x llama-tools/.venv/bin/pip ]] \
    || die "venv reported success but llama-tools/.venv/bin/pip is missing or not executable" \
           "${EXIT_ENV}"
fi

run_classified "pip install --upgrade pip" "${EXIT_ENV}" \
  llama-tools/.venv/bin/pip install -q --upgrade pip
# torch is intentionally NOT installed: it comes from the template's CUDA build.
run_classified "pip install -r requirements-probe.txt" "${EXIT_ENV}" \
  llama-tools/.venv/bin/pip install -q -r llama-tools/requirements-probe.txt

# ---------------------------------------------------------------------------
# STEP 6 — preflight. Everything that must be true BEFORE money is spent on
# inference. Each check corresponds to a way a run has failed, or could fail,
# after the meter was already running.
# ---------------------------------------------------------------------------
echo
echo "STEP 6 — preflight (fail closed)"

# GNU timeout with --kill-after: the wall-clock spend cap is enforced by it.
# Without it launch_probe.sh refuses to start rather than run uncapped.
if [[ "${dry_run}" -eq 0 ]]; then
  timeout_bin=""
  for c in timeout gtimeout; do
    if command -v "$c" >/dev/null; then
      timeout_bin="$c"
      break
    fi
  done
  [[ -n "${timeout_bin}" ]] || die "GNU timeout not found; the spend cap cannot be enforced" \
    "${EXIT_ENV}"
  "${timeout_bin}" --kill-after=1 1 true \
    || die "${timeout_bin} does not support --kill-after" "${EXIT_ENV}"
  echo "  timeout: ${timeout_bin} with --kill-after OK"
else
  announce "command -v timeout && timeout --kill-after=1 1 true"
fi

if [[ "${dry_run}" -eq 0 ]]; then
  # `|| exit 68` was a bare literal duplicating EXIT_ENV, and it exited with no
  # message of its own — the operator saw only Python's traceback and had to
  # infer which preflight had failed. It now names the check and uses the
  # constant, so renumbering EXIT_ENV cannot leave a stale 68 behind here.
  env_preflight_status=0
  llama-tools/.venv/bin/python - "${out_root}" <<'PY' || env_preflight_status=$?
import json, sys
from importlib.metadata import version

out_root = sys.argv[1]
sys.path.insert(0, "llama-tools")

from eval.environment_fingerprint import collect_locale_provenance

expected = {
    "transformers": "5.14.1",
    "peft": "0.19.1",
    "accelerate": "1.14.0",
    "huggingface-hub": "1.24.0",
}
bad = {p: (version(p), want) for p, want in expected.items() if version(p) != want}
assert not bad, f"probe version tuple mismatch: {bad}"

import accelerate, peft, torch, transformers  # imports must actually work
assert torch.cuda.is_available(), "no CUDA device visible"

locale_provenance = collect_locale_provenance()
fingerprint = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,          # from the image, not pinned by us
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "accelerate": accelerate.__version__,
    # Locale is execution provenance, not decoration. Ask the shell's `locale`
    # resolver rather than CPython: Python can report LC_COLLATE=C even while
    # Bash is using en_US.UTF-8 from the same environment.
    "locale": locale_provenance,
}
with open(f"{out_root}/env_fingerprint.json", "w", encoding="utf-8") as f:
    json.dump(fingerprint, f, indent=2, sort_keys=True)
print("  versions + imports + CUDA OK:", fingerprint["gpu"], "| CUDA", fingerprint["cuda"])
PY
  [[ "${env_preflight_status}" -eq 0 ]] \
    || die "environment preflight failed (exit ${env_preflight_status}): version tuple, imports, or CUDA availability — see the traceback above" \
           "${EXIT_ENV}"
  # The fingerprint is the artifact that makes this run comparable to any other,
  # WITHOUT a later reader having to go excavating. The mining pilot is the
  # counterexample: its fingerprint was never written, and the versions had to be
  # reconstructed from console output pasted into chat months later. The
  # comparison was possible; it just depended on someone remembering where to
  # look. Asserting the file landed is what makes the next run's provenance
  # self-contained, and that gap was silent at the time it was created.
  assert_file "${out_root}/env_fingerprint.json" "environment fingerprint" "${EXIT_ENV}"
else
  announce "python -c 'assert exact version tuple, imports, torch.cuda.is_available()'"
fi

# HF access to the gated base model and the private SFT adapter, checked before
# a 16GB download is attempted on billed time.
if [[ "${dry_run}" -eq 0 ]]; then
  hf_preflight_status=0
  llama-tools/.venv/bin/python - <<'PY' || hf_preflight_status=$?
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.repo_info("meta-llama/Llama-3.1-8B-Instruct",
              revision="0e9e39f249a16976918f6564b8830bc894c89659")
api.repo_info("centuriandip/llama-3.1-8b-tools-sft",
              revision="b6f4da479f8c6fc044ee8b802a92f47780f970c5")
print("  HF access OK (gated base + private adapter)")
PY
  [[ "${hf_preflight_status}" -eq 0 ]] \
    || die "HF access preflight failed (exit ${hf_preflight_status}): the gated base or the private adapter is not reachable with this HF_TOKEN — see the traceback above" \
           "${EXIT_ENV}"
else
  announce "python -c 'HfApi().repo_info(base@rev); repo_info(sft-adapter@rev)'"
fi

# ---------------------------------------------------------------------------
# STEP 7 — record environment evidence directly into the persistent root.
# ---------------------------------------------------------------------------
echo
echo "STEP 7 — environment evidence -> ${out_root}"
locale_gap_reason=""
if [[ "${dry_run}" -eq 0 ]]; then
  pip_freeze_status=0
  llama-tools/.venv/bin/pip freeze > "${out_root}/pip_freeze.txt" || pip_freeze_status=$?
  [[ "${pip_freeze_status}" -eq 0 ]] \
    || die "pip freeze failed (exit ${pip_freeze_status}); the dependency receipt for this run cannot be written" \
           "${EXIT_ENV}"

  # This block used to read:
  #
  #   nvidia-smi ... > gpu.txt 2>/dev/null || echo "nvidia-smi unavailable" > gpu.txt
  #
  # which is the fail-open pattern in its purest form. It discarded the reason,
  # then WROTE AN ARTIFACT ASSERTING a reason it had just thrown away — and the
  # run continued to paid inference as if the environment had been recorded.
  # "unavailable" is also flatly inconsistent with STEP 6, which has already
  # asserted torch.cuda.is_available() on this same host: if the CUDA runtime
  # can see a device and nvidia-smi cannot, something is wrong with the pod that
  # an operator must know about BEFORE spending, not discover in a postmortem.
  # Stderr is preserved as evidence and the run stops.
  nvidia_smi_status=0
  nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv > "${out_root}/gpu.txt" 2> "${out_root}/gpu.stderr.txt" \
    || nvidia_smi_status=$?
  if [[ "${nvidia_smi_status}" -ne 0 ]]; then
    echo "--- nvidia-smi stderr ---" >&2
    cat "${out_root}/gpu.stderr.txt" >&2
    die "nvidia-smi failed (exit ${nvidia_smi_status}) on a host where STEP 6 already asserted torch.cuda.is_available(); stderr preserved at ${out_root}/gpu.stderr.txt" \
        "${EXIT_ENV}"
  fi
  rm -f "${out_root}/gpu.stderr.txt"

  echo "${image_tag}" > "${out_root}/image_tag.txt"
  echo "${auto_terminate_set}" > "${out_root}/auto_terminate_attestation.txt"
  echo "${commit}" > "${out_root}/reviewed_commit.txt"

  # Explicit inventory assertion. Every file below is something a later reader
  # needs in order to say what this run's environment WAS; a missing one is not
  # a cosmetic gap, it is the difference between a measured claim and a guess.
  # Checked here rather than trusted, because each was written by a separate
  # command and `set -e` only proves those commands returned zero.
  assert_file "${out_root}/pip_freeze.txt"                 "pip freeze receipt"        "${EXIT_ENV}"
  assert_file "${out_root}/gpu.txt"                        "GPU receipt"               "${EXIT_ENV}"
  assert_file "${out_root}/image_tag.txt"                  "image tag receipt"         "${EXIT_ENV}"
  assert_file "${out_root}/auto_terminate_attestation.txt" "auto-terminate attestation" "${EXIT_ENV}"
  assert_file "${out_root}/reviewed_commit.txt"            "reviewed-commit receipt"   "${EXIT_ENV}"
  assert_file "${out_root}/env_fingerprint.json"           "environment fingerprint"   "${EXIT_ENV}"
  assert_file "${out_root}/bundle_sha256.txt"              "bundle hash receipt"       "${EXIT_ENV}"

  # Locale collection is best-effort telemetry, not a correctness gate. The
  # monitor's JSON escaping is locale-independent now, so a minimal image that
  # lacks the `locale` utility costs one provenance field but does not invalidate
  # the run. Keep that gap visible and durable without burning a billed
  # bootstrap. An unknown/corrupt status still fails loudly.
  locale_gap_status=0
  locale_gap_reason="$(
    llama-tools/.venv/bin/python - "${out_root}/env_fingerprint.json" <<'PY'
import json, sys

with open(sys.argv[1], encoding="utf-8") as handle:
    provenance = json.load(handle)["locale"]
status = provenance.get("status")
if status == "unavailable":
    print(str(provenance.get("reason") or "reason missing"))
elif status != "ok":
    raise RuntimeError(f"unknown locale provenance status: {status!r}")
PY
  )" || locale_gap_status=$?
  [[ "${locale_gap_status}" -eq 0 ]] \
    || die "locale provenance status could not be read from env_fingerprint.json (exit ${locale_gap_status})" \
           "${EXIT_ENV}"
  if [[ -n "${locale_gap_reason}" ]]; then
    formatted_locale_gap="${locale_gap_reason//$'\n'/$'\n      '}"
    echo "  EVIDENCE GAP — locale provenance unavailable; unmeasured, not defaulted:"
    echo "    - ${formatted_locale_gap}"
  fi
  echo "  all 7 environment receipts asserted present and non-empty"
  ls -1 "${out_root}"
else
  announce "pip freeze / nvidia-smi / image tag / attestation -> ${out_root}"
  announce "assert all 7 environment receipts exist and are non-empty"
fi

cat <<EOF

=====================================================================
BOOTSTRAP COMPLETE — pod is launch-ready at ${commit}

Billing has been running since pod start. Everything above was billed.

Next (the only remaining step, and it is the paid one):

  cd llama-tools
  tmux new-session -d -s probe \\
    "bash scripts/launch_probe.sh \\
       --commit ${commit} \\
       --provider-deadline-epoch <EPOCH_FROM_PROVIDER_DEADLINE> \\
       --deadline-epoch <PROVIDER_DEADLINE_MINUS_SHUTDOWN_RESERVE> \\
       --out-root ${out_root} 2>&1 | tee ${out_root}/probe.log"
  tmux attach -t probe      # optional; detaching does not stop the run

Detached session + tee: the run survives a dropped SSH connection and its
stdout/stderr lands in the persistent root as durable evidence.
Derive --deadline-epoch immediately before launch by subtracting the shutdown
reserve from the provider deadline; see docs/probe-bootstrap.md.
=====================================================================
EOF

exit "${EXIT_OK}"
