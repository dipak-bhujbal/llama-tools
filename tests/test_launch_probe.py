"""Tests for scripts/launch_probe.sh.

These tests only ever invoke the script with --dry-run, so they never touch
git state, the network, or spawn a real generation run — the script's
--dry-run contract (print every command it would run, execute nothing) is
exactly what makes it possible to test a launch procedure without a pod.

Every test shells out via `bash scripts/launch_probe.sh ...` and asserts on
exit code / stdout+stderr text, mirroring how a reviewer would actually
exercise the script.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "launch_probe.sh"

# A syntactically valid 40-char hex SHA. It does not need to exist as a real
# commit for --dry-run: the script never calls `git checkout` for real in
# dry-run mode, it only prints the command it would run.
VALID_SHA = "1f0850103660ab46dc489a4c91280190b4da6620"

REQUIRED_FLAGS = {
    "--commit": VALID_SHA,
    "--usd-cap": "2.50",
    "--usd-per-hour": "0.44",
    "--out-root": "/tmp/launch_probe_test_out_root",
}


def run_script(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke the script with `bash`, capturing combined-ish stdout/stderr."""
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )


def full_args(overrides: dict[str, str] | None = None, omit: set[str] | None = None,
              extra: list[str] | None = None) -> list[str]:
    """Build a full valid arg list, optionally overriding/omitting flags."""
    overrides = overrides or {}
    omit = omit or set()
    flags = {**REQUIRED_FLAGS, **overrides}
    args: list[str] = []
    for flag, value in flags.items():
        if flag in omit:
            continue
        args.extend([flag, value])
    args.extend(extra or [])
    return args


def combined_output(result: subprocess.CompletedProcess) -> str:
    return result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Missing required flags: each must exit non-zero and name the missing flag.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_flag", sorted(REQUIRED_FLAGS))
def test_missing_required_flag_exits_nonzero_and_names_flag(missing_flag: str) -> None:
    result = run_script(full_args(omit={missing_flag}, extra=["--dry-run"]))
    assert result.returncode != 0, combined_output(result)
    assert missing_flag in combined_output(result), combined_output(result)


def test_missing_all_flags_exits_nonzero_and_names_all() -> None:
    result = run_script(["--dry-run"])
    assert result.returncode != 0
    output = combined_output(result)
    for flag in REQUIRED_FLAGS:
        assert flag in output, output


# ---------------------------------------------------------------------------
# Commit SHA validation: must be rejected unless it is exactly 40 hex chars.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_commit",
    [
        "deadbeef",  # too short
        "g" * 40,  # 40 chars but not hex
        VALID_SHA + "0",  # 41 chars
        VALID_SHA[:-1],  # 39 chars
        "",  # empty (also caught by missing-flag path, but check explicitly)
    ],
)
def test_non_full_hex_commit_is_rejected(bad_commit: str) -> None:
    if bad_commit == "":
        # An empty value after --commit is indistinguishable from "no value
        # supplied" by the script's own require_value guard; that path is
        # covered by the missing-flag tests, so skip it here to avoid
        # asserting on the wrong error message.
        pytest.skip("empty commit value is handled by the missing-value guard")
    result = run_script(full_args(overrides={"--commit": bad_commit}, extra=["--dry-run"]))
    assert result.returncode != 0, combined_output(result)
    assert "--commit" in combined_output(result), combined_output(result)


def test_valid_commit_passes_validation() -> None:
    result = run_script(full_args(extra=["--dry-run"]))
    assert result.returncode == 0, combined_output(result)


# ---------------------------------------------------------------------------
# --dry-run content and ordering contract.
# ---------------------------------------------------------------------------


def test_dry_run_exits_zero() -> None:
    result = run_script(full_args(extra=["--dry-run"]))
    assert result.returncode == 0, combined_output(result)


def test_dry_run_contains_detached_checkout_of_given_sha() -> None:
    result = run_script(full_args(extra=["--dry-run"]))
    output = combined_output(result)
    assert "checkout --detach" in output
    assert VALID_SHA in output


def test_dry_run_does_not_contain_git_pull() -> None:
    result = run_script(full_args(extra=["--dry-run"]))
    output = combined_output(result)
    assert "git pull" not in output


def test_dry_run_contains_acquire_and_verify_and_both_generation_commands() -> None:
    result = run_script(full_args(extra=["--dry-run"]))
    output = combined_output(result)

    # Acquire: fetch_pinned_bfcl.py WITHOUT --verify-only.
    acquire_lines = [
        line for line in output.splitlines()
        if "fetch_pinned_bfcl.py" in line and "--verify-only" not in line
    ]
    assert acquire_lines, output

    # Verify: fetch_pinned_bfcl.py WITH --verify-only (appears twice: once
    # before each paid generation command).
    verify_lines = [line for line in output.splitlines() if "--verify-only" in line]
    assert len(verify_lines) >= 2, output

    # Both paid generation commands, verbatim flags preserved.
    assert "--category multiple" in output
    assert "--category simple_python" in output
    assert "bfcl_simple.py" in output
    assert "--sft-adapter-revision b6f4da479f8c6fc044ee8b802a92f47780f970c5" in output
    assert "--base-revision 0e9e39f249a16976918f6564b8830bc894c89659" in output
    assert "study2_probe_multiple" in output
    assert "study2_probe_simple_python" in output


def test_dry_run_orders_acquire_before_verify_before_first_generation() -> None:
    result = run_script(full_args(extra=["--dry-run"]))
    output = combined_output(result)

    acquire_index = next(
        i for i, line in enumerate(output.splitlines())
        if "fetch_pinned_bfcl.py" in line and "--verify-only" not in line
    )
    verify_index = next(
        i for i, line in enumerate(output.splitlines()) if "--verify-only" in line
    )
    first_generation_index = next(
        i for i, line in enumerate(output.splitlines())
        if "bfcl_simple.py" in line and "--category multiple" in line
    )
    second_generation_index = next(
        i for i, line in enumerate(output.splitlines())
        if "bfcl_simple.py" in line and "--category simple_python" in line
    )

    assert acquire_index < verify_index < first_generation_index < second_generation_index, output


# ---------------------------------------------------------------------------
# Budget derivation: printed, and scales the right direction.
# ---------------------------------------------------------------------------

_MAX_SECONDS_RE = re.compile(r"per_command_max_seconds=(\d+)")


def _derived_per_command_seconds(usd_cap: str, usd_per_hour: str) -> int:
    result = run_script(
        full_args(
            overrides={"--usd-cap": usd_cap, "--usd-per-hour": usd_per_hour},
            extra=["--dry-run"],
        )
    )
    output = combined_output(result)
    assert result.returncode == 0, output
    match = _MAX_SECONDS_RE.search(output)
    assert match, output
    return int(match.group(1))


def test_derived_budget_is_printed() -> None:
    result = run_script(full_args(extra=["--dry-run"]))
    output = combined_output(result)
    assert "BUDGET" in output
    assert _MAX_SECONDS_RE.search(output), output


def test_derived_budget_scales_correctly_with_hourly_rate() -> None:
    # A cheaper hourly rate affords more wall-clock time for the same
    # dollar cap, so the derived --max-seconds budget must be strictly
    # larger for --usd-per-hour 0.44 than for --usd-per-hour 1.00.
    cheap_rate_seconds = _derived_per_command_seconds("2.50", "0.44")
    expensive_rate_seconds = _derived_per_command_seconds("2.50", "1.00")
    assert cheap_rate_seconds > expensive_rate_seconds


def test_zero_or_negative_usd_cap_is_rejected() -> None:
    result = run_script(full_args(overrides={"--usd-cap": "0"}, extra=["--dry-run"]))
    assert result.returncode != 0
    assert "--usd-cap" in combined_output(result)

    result = run_script(full_args(overrides={"--usd-cap": "-1"}, extra=["--dry-run"]))
    assert result.returncode != 0
    assert "--usd-cap" in combined_output(result)


def test_zero_or_negative_usd_per_hour_is_rejected() -> None:
    result = run_script(full_args(overrides={"--usd-per-hour": "0"}, extra=["--dry-run"]))
    assert result.returncode != 0
    assert "--usd-per-hour" in combined_output(result)


# ---------------------------------------------------------------------------
# --dry-run must not create the out-root directory or modify git state.
# ---------------------------------------------------------------------------


def test_dry_run_does_not_create_out_root_directory(tmp_path: Path) -> None:
    out_root = tmp_path / "probe_out_root_should_not_exist"
    assert not out_root.exists()

    result = run_script(
        full_args(overrides={"--out-root": str(out_root)}, extra=["--dry-run"])
    )
    assert result.returncode == 0, combined_output(result)
    assert not out_root.exists(), "dry-run must not create --out-root"


def test_dry_run_does_not_modify_git_state() -> None:
    head_before = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status_before = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    result = run_script(full_args(extra=["--dry-run"]))
    assert result.returncode == 0, combined_output(result)

    head_after = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status_after = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert head_before == head_after, "dry-run must not move HEAD"
    assert status_before == status_after, "dry-run must not dirty the working tree"


# ---------------------------------------------------------------------------
# STOP-THE-POD reminder and artifact paths.
# ---------------------------------------------------------------------------


def test_output_includes_stop_the_pod_reminder_and_artifact_paths() -> None:
    result = run_script(full_args(extra=["--dry-run"]))
    output = combined_output(result)
    assert "STOP THE POD" in output
    assert "study2_probe_multiple/generations.jsonl" in output
    assert "study2_probe_simple_python/generations.jsonl" in output


def test_preflight_warns_in_dry_run_when_timeout_binary_is_absent(tmp_path) -> None:
    """The approved spend cap is enforced by `timeout`. A host without it must
    be told, because a plan that prints fine here would refuse to start on a
    pod. Dry run warns rather than fails so the plan stays reviewable off-pod.
    """
    # Stripping PATH down to a stub dir is not viable: the script legitimately
    # needs dirname/awk too, so a minimal PATH fails for the wrong reason.
    # Instead branch on what this host actually has, so the assertion is true
    # on both a coreutils Linux pod and a bare macOS laptop.
    has_timeout = bool(shutil.which("timeout") or shutil.which("gtimeout"))
    result = run_script([*full_args(), "--dry-run"])
    assert result.returncode == 0, combined_output(result)
    out = combined_output(result)

    if has_timeout:
        assert "PREFLIGHT WARNING" not in out, out
        assert "PREFLIGHT: wall-clock enforcement via" in out, out
    else:
        assert "PREFLIGHT WARNING" in out, out
        assert "REFUSE to start" in out, out


def test_preflight_is_announced_before_any_git_mutation() -> None:
    """Ordering matters: discovering the missing binary after `checkout
    --detach` would strand the repo on a detached HEAD and bill a pod for a
    download that can never be used."""
    result = run_script([*full_args(), "--dry-run"])
    out = combined_output(result)
    assert "PREFLIGHT" in out
    assert out.index("PREFLIGHT") < out.index("checkout --detach")
