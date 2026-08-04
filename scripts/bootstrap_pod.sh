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
# BILLED time and all consume the same $2.50 lifecycle budget as generation.
# There is no "$0 until launch" phase. That is why --auto-terminate-set must
# be acknowledged before this script does anything else: the provider-side
# deadline has to already exist by the time the pod is running.
#
# Usage:
#   scripts/bootstrap_pod.sh \
#     --bundle /workspace/llama-tools.bundle \
#     --bundle-sha256-file /workspace/llama-tools.bundle.sha256 \
#     --commit <FULL_40_CHAR_SHA> \
#     --out-root /workspace/persist/study2 \
#     --auto-terminate-set "2026-08-04T23:00:00Z@0.44usd/hr" \
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

die() { echo "ERROR: $*" >&2; exit "${2:-$EXIT_USAGE}"; }

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

run() {
  announce "$@"
  [[ "${dry_run}" -eq 1 ]] && return 0
  "$@"
}

# ---------------------------------------------------------------------------
# STEP 0 — provider spend cap. FIRST, because billing is already running.
#
# A process that has been SIGKILLed cannot stop its own billing, and owner
# approval to spend at most $2.50 authorises an amount; it does not enforce
# one. Only the provider-side deadline does. This script cannot set it (no
# account credentials, by design) and cannot verify it from inside the pod, so
# it requires an explicit acknowledgement string and records it as evidence.
# That is deliberately a human attestation, not a simulated check.
# ---------------------------------------------------------------------------
echo "====================================================================="
echo "STEP 0 — provider auto-termination"
if [[ ! "${auto_terminate_set}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z@.+ ]]; then
  echo "ERROR: --auto-terminate-set must look like" >&2
  echo "       <ISO8601-deadline-Z>@<rate>, e.g. 2026-08-04T23:00:00Z@0.44usd/hr" >&2
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
run mkdir -p "${out_root}"
if [[ "${dry_run}" -eq 0 ]]; then
  touch "${out_root}/.write_probe" 2>/dev/null \
    || die "--out-root ${out_root} is not writable" "${EXIT_ENV}"
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
  expected_sha="$(tr -d '[:space:]' < "${bundle_sha_file}" | cut -c1-64)"
  if command -v sha256sum >/dev/null 2>&1; then
    actual_sha="$(sha256sum "${bundle}" | awk '{print $1}')"
  else
    actual_sha="$(shasum -a 256 "${bundle}" | awk '{print $1}')"
  fi
  [[ "${actual_sha}" == "${expected_sha}" ]] \
    || die "bundle hash mismatch: got ${actual_sha}, expected ${expected_sha}" "${EXIT_BUNDLE}"
  echo "  bundle sha256 verified: ${actual_sha}"
  echo "${actual_sha}" > "${out_root}/bundle_sha256.txt"
else
  announce sha256sum "${bundle}" "# compared against ${bundle_sha_file}"
fi

# ---------------------------------------------------------------------------
# STEP 4 — clone and detach at the exact reviewed SHA.
# ---------------------------------------------------------------------------
echo
echo "STEP 4 — clone at the reviewed commit"
run git clone "${bundle}" llama-tools
run git -C llama-tools checkout --detach "${commit}"
if [[ "${dry_run}" -eq 0 ]]; then
  head_sha="$(git -C llama-tools rev-parse HEAD)"
  [[ "${head_sha}" == "${commit}" ]] \
    || die "HEAD is ${head_sha}, expected ${commit}" "${EXIT_GIT}"
  [[ -z "$(git -C llama-tools status --porcelain)" ]] \
    || die "working tree is dirty immediately after clone" "${EXIT_GIT}"
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
run python3 -m venv --system-site-packages llama-tools/.venv
run llama-tools/.venv/bin/pip install -q --upgrade pip
# torch is intentionally NOT installed: it comes from the template's CUDA build.
run llama-tools/.venv/bin/pip install -q -r llama-tools/requirements-probe.txt

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
    command -v "$c" >/dev/null 2>&1 && { timeout_bin="$c"; break; }
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
  llama-tools/.venv/bin/python - "${out_root}" <<'PY' || exit 68
import json, sys
from importlib.metadata import version

out_root = sys.argv[1]
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

fingerprint = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,          # from the image, not pinned by us
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "accelerate": accelerate.__version__,
}
with open(f"{out_root}/env_fingerprint.json", "w") as f:
    json.dump(fingerprint, f, indent=2, sort_keys=True)
print("  versions + imports + CUDA OK:", fingerprint["gpu"], "| CUDA", fingerprint["cuda"])
PY
else
  announce "python -c 'assert exact version tuple, imports, torch.cuda.is_available()'"
fi

# HF access to the gated base model and the private SFT adapter, checked before
# a 16GB download is attempted on billed time.
if [[ "${dry_run}" -eq 0 ]]; then
  llama-tools/.venv/bin/python - <<'PY' || exit 68
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.repo_info("meta-llama/Llama-3.1-8B-Instruct",
              revision="0e9e39f249a16976918f6564b8830bc894c89659")
api.repo_info("centuriandip/llama-3.1-8b-tools-sft",
              revision="b6f4da479f8c6fc044ee8b802a92f47780f970c5")
print("  HF access OK (gated base + private adapter)")
PY
else
  announce "python -c 'HfApi().repo_info(base@rev); repo_info(sft-adapter@rev)'"
fi

# ---------------------------------------------------------------------------
# STEP 7 — record environment evidence directly into the persistent root.
# ---------------------------------------------------------------------------
echo
echo "STEP 7 — environment evidence -> ${out_root}"
if [[ "${dry_run}" -eq 0 ]]; then
  llama-tools/.venv/bin/pip freeze > "${out_root}/pip_freeze.txt"
  nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv > "${out_root}/gpu.txt" 2>/dev/null || echo "nvidia-smi unavailable" \
    > "${out_root}/gpu.txt"
  echo "${image_tag}" > "${out_root}/image_tag.txt"
  echo "${auto_terminate_set}" > "${out_root}/auto_terminate_attestation.txt"
  echo "${commit}" > "${out_root}/reviewed_commit.txt"
  ls -1 "${out_root}"
else
  announce "pip freeze / nvidia-smi / image tag / attestation -> ${out_root}"
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
       --usd-cap 2.50 \\
       --usd-per-hour <ACTUAL_RATE> \\
       --out-root ${out_root} 2>&1 | tee ${out_root}/probe.log"
  tmux attach -t probe      # optional; detaching does not stop the run

Detached session + tee: the run survives a dropped SSH connection and its
stdout/stderr lands in the persistent root as durable evidence.
=====================================================================
EOF

exit "${EXIT_OK}"
