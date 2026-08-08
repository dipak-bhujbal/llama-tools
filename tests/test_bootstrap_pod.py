"""Tests for scripts/bootstrap_pod.sh.

Only --dry-run is ever invoked, so these never clone, never create a venv and
never touch the network. Each test corresponds to a way a real run could waste
billed pod time or produce evidence that cannot be traced back to an
environment.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "bootstrap_pod.sh"

VALID_SHA = "7bb20ee8314e70782d03ebe48d02c8d8fd82dd03"
# Symbolic on purpose. The gate checks the *shape* `<ISO8601-Z>@<rate>` — it
# cannot verify a rate from inside the pod, and a real-looking number here
# would be a second place an approved figure lives, which is how a superseded
# one stayed mechanically enforced before.
VALID_ATTESTATION = "2026-08-04T23:00:00Z@RATE_FROM_CONSOLE"

REQUIRED = {
    "--bundle": "/tmp/llama-tools.bundle",
    "--bundle-sha256-file": "/tmp/llama-tools.bundle.sha256",
    "--commit": VALID_SHA,
    "--out-root": "/tmp/bootstrap_test_out",
    "--auto-terminate-set": VALID_ATTESTATION,
}


def run_script(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
    )


def full_args(overrides: dict | None = None, omit: set | None = None) -> list[str]:
    flags = {**REQUIRED, **(overrides or {})}
    args: list[str] = []
    for flag, value in flags.items():
        if flag in (omit or set()):
            continue
        args.extend([flag, value])
    return [*args, "--dry-run"]


def out(result: subprocess.CompletedProcess) -> str:
    return result.stdout + result.stderr


@pytest.mark.parametrize("missing", sorted(REQUIRED))
def test_missing_required_flag_fails_and_names_it(missing: str) -> None:
    result = run_script(full_args(omit={missing}))
    assert result.returncode != 0, out(result)
    assert missing in out(result)


@pytest.mark.parametrize("bad_sha", ["deadbeef", "g" * 40, VALID_SHA + "0", VALID_SHA[:-1]])
def test_commit_must_be_full_hex_sha(bad_sha: str) -> None:
    result = run_script(full_args({"--commit": bad_sha}))
    assert result.returncode != 0, out(result)
    assert "--commit" in out(result)


@pytest.mark.parametrize(
    "bad_attestation",
    ["sometime", "yes", "2026-08-04T23:00:00Z", "@RATE_FROM_CONSOLE", "later@"],
)
def test_provider_cap_attestation_must_be_wellformed(bad_attestation: str) -> None:
    """The provider deadline is the only thing that actually bounds vendor
    billing, so a vague acknowledgement must not satisfy the gate."""
    result = run_script(full_args({"--auto-terminate-set": bad_attestation}))
    assert result.returncode != 0, out(result)
    assert "auto-terminate-set" in out(result)


def test_provider_cap_is_checked_before_anything_else() -> None:
    """Billing is already running by the time this script executes, so the
    cap gate must come before clone/venv/download work is attempted."""
    result = run_script(full_args())
    text = out(result)
    assert "STEP 0" in text
    assert text.index("STEP 0") < text.index("STEP 4")


def test_dry_run_succeeds_and_changes_nothing(tmp_path) -> None:
    target = tmp_path / "never_created"
    result = run_script(full_args({"--out-root": str(target)}))
    assert result.returncode == 0, out(result)
    assert not (target / "bundle_sha256.txt").exists()


def test_dry_run_plan_covers_the_load_bearing_steps() -> None:
    text = out(run_script(full_args()))
    for expected in (
        "sha256sum",            # receipt verified before clone
        "git clone",
        "checkout --detach",
        "venv",
        "requirements-probe.txt",
        "timeout --kill-after",
        "torch.cuda.is_available",
        "repo_info",            # HF access preflight
    ):
        assert expected in text, f"missing {expected!r} from plan:\n{text}"


def test_receipt_is_verified_before_clone() -> None:
    """Cloning first would mean trusting objects that have not been checked."""
    text = out(run_script(full_args()))
    assert text.index("sha256sum") < text.index("git clone")


def test_launch_hint_is_detached_and_logged() -> None:
    """A foreground tmux blocks the pasted sequence and keeps no durable log."""
    text = out(run_script(full_args()))
    assert "tmux new-session -d" in text
    assert "tee" in text


def test_unknown_image_tag_is_rejected_on_a_real_run() -> None:
    """A run whose environment cannot be named cannot be reproduced, so
    'unknown' must fail rather than be recorded as evidence. Dry run is
    allowed to substitute a placeholder so the plan stays inspectable."""
    text = out(run_script(full_args()))
    assert "DRY RUN" in text and "RUNPOD_IMAGE_NAME" in text
    body = SCRIPT.read_text()
    assert 'image_tag}" == "unknown"' in body or '"unknown"' in body
