"""Tests for scripts/probe_liveness.sh.

The monitor this replaces alerted on log staleness, which is wrong in both
directions: it cried wolf during a five-minute model load and stayed silent when
the process was killed outright. So the two load-bearing tests here are

  * a live process with a stale log is NOT an alert, and
  * a dead process with no terminal record IS one.

Everything runs locally with real (short-lived) processes and temp files.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "probe_liveness.sh"

EXIT_ALIVE = 0
EXIT_USAGE = 64
EXIT_COMPLETED_OK = 70
EXIT_COMPLETED_FAILED = 71
EXIT_DIED_HARD = 72


def run_monitor(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
    )


@pytest.fixture
def dead_pid() -> int:
    """A PID that is definitively gone: spawned, exited, and reaped."""
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "probe.log", tmp_path / "status.json"


def read_status(status_file: Path) -> dict:
    return json.loads(status_file.read_text(encoding="utf-8"))


# --- usage ------------------------------------------------------------------
@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--log", "/tmp/x"],
        ["--log", "/tmp/x", "--status-file", "/tmp/y"],  # no --pid or --tmux-session
        ["--log", "/tmp/x", "--status-file", "/tmp/y", "--pid", "not-a-number"],
    ],
)
def test_incomplete_invocation_exits_usage(args: list[str]) -> None:
    assert run_monitor([*args, "--once"]).returncode == EXIT_USAGE


def test_missing_watch_target_names_what_is_missing() -> None:
    result = run_monitor(["--log", "/tmp/x", "--status-file", "/tmp/y", "--once"])
    assert "--pid or --tmux-session" in result.stderr


# --- assertion 1: the process is alive --------------------------------------
def test_live_process_reports_running(paths) -> None:
    log, status = paths
    log.write_text("Loading base model...\n", encoding="utf-8")
    result = run_monitor(
        ["--log", str(log), "--status-file", str(status), "--pid", str(os.getpid()), "--once"]
    )
    assert result.returncode == EXIT_ALIVE
    record = read_status(status)
    assert record["state"] == "running"
    assert record["pid_alive"] is True
    assert record["alert"] is False


def test_a_stale_log_under_a_live_process_is_not_an_alert(paths) -> None:
    """The regression test for the old monitor's central defect. An 8B model
    load writes nothing for minutes; alerting on that trained the operator to
    ignore the alerts."""
    log, status = paths
    log.write_text("Loading base model...\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(log, (old, old))

    result = run_monitor(
        ["--log", str(log), "--status-file", str(status), "--pid", str(os.getpid()), "--once"]
    )
    assert result.returncode == EXIT_ALIVE
    record = read_status(status)
    assert record["state"] == "running"
    assert record["alert"] is False
    # Age is still reported — it is useful context — but flagged as non-alerting.
    assert record["log_age_seconds"] >= 3500
    assert record["log_age_is_not_an_alert_signal"] is True


def test_error_markers_under_a_live_process_are_reported_without_killing_it(paths) -> None:
    log, status = paths
    log.write_text(
        "step ok\nRuntimeError: CUDA error: an illegal memory access was encountered\n",
        encoding="utf-8",
    )
    result = run_monitor(
        ["--log", str(log), "--status-file", str(status), "--pid", str(os.getpid()), "--once"]
    )
    assert result.returncode == EXIT_ALIVE  # a monitor never terminates a paid run
    record = read_status(status)
    assert record["error_markers_seen"] >= 2
    assert "CUDA error" in record["error_marker_names"]
    assert Path(str(status) + ".errors.txt").exists()


# --- assertion 2/3: a terminal record with an exit code ---------------------
def test_dead_process_with_clean_exit_record(paths, dead_pid: int) -> None:
    log, status = paths
    log.write_text(
        "RUN COMPLETE — elapsed 900s\n"
        f"PROBE_EXIT_RECORD pid={dead_pid} exit=0 elapsed=900s completed_all_steps=1 dry_run=0\n",
        encoding="utf-8",
    )
    result = run_monitor(
        ["--log", str(log), "--status-file", str(status), "--pid", str(dead_pid), "--once"]
    )
    assert result.returncode == EXIT_COMPLETED_OK
    record = read_status(status)
    assert record["state"] == "exited"
    assert record["exit_code"] == 0
    assert record["alert"] is False


def test_dead_process_with_failure_exit_record_carries_the_code(paths, dead_pid: int) -> None:
    log, status = paths
    log.write_text(
        "RUN DID NOT COMPLETE — exit 68, elapsed 64s\n"
        f"PROBE_EXIT_RECORD pid={dead_pid} exit=68 elapsed=64s completed_all_steps=0 dry_run=0\n",
        encoding="utf-8",
    )
    result = run_monitor(
        ["--log", str(log), "--status-file", str(status), "--pid", str(dead_pid), "--once"]
    )
    assert result.returncode == EXIT_COMPLETED_FAILED
    record = read_status(status)
    assert record["state"] == "exited"
    assert record["exit_code"] == 68  # EXIT_GENERATION_FAILED, the §0 probe's own code
    assert record["alert"] is True


def test_prose_footer_alone_still_yields_an_exit_code(paths, dead_pid: int) -> None:
    """Logs from a launcher predating the machine-readable record must still be
    readable; the fallback is weaker evidence but not no evidence."""
    log, status = paths
    log.write_text("RUN DID NOT COMPLETE — exit 67, elapsed 12s\n", encoding="utf-8")
    result = run_monitor(
        ["--log", str(log), "--status-file", str(status), "--pid", str(dead_pid), "--once"]
    )
    assert result.returncode == EXIT_COMPLETED_FAILED
    assert read_status(status)["exit_code"] == 67


# --- the alert: died hard ---------------------------------------------------
def test_dead_process_with_no_terminal_record_is_the_alert(paths, dead_pid: int) -> None:
    log, status = paths
    log.write_text("Loading base model...\nGenerating with base\n", encoding="utf-8")
    result = run_monitor(
        ["--log", str(log), "--status-file", str(status), "--pid", str(dead_pid), "--once"]
    )
    assert result.returncode == EXIT_DIED_HARD
    record = read_status(status)
    assert record["state"] == "died_hard"
    assert record["exit_code"] is None  # unknown, and said so rather than guessed
    assert record["alert"] is True
    assert "DIED HARD" in result.stderr
    # The operator's next action, because a killed process cannot stop its meter.
    assert "CONFIRM BILLING STOPPED" in result.stderr


def test_a_terminal_record_from_a_different_pid_does_not_count_as_ours(paths, dead_pid: int) -> None:
    """A log appended by two consecutive runs would otherwise let run A's clean
    exit be reported as run B's outcome — stale evidence read as live."""
    log, status = paths
    other = dead_pid + 1 if dead_pid > 1 else 99999
    log.write_text(
        "RUN COMPLETE — elapsed 900s\n"
        f"PROBE_EXIT_RECORD pid={other} exit=0 elapsed=900s completed_all_steps=1 dry_run=0\n",
        encoding="utf-8",
    )
    result = run_monitor(
        ["--log", str(log), "--status-file", str(status), "--pid", str(dead_pid), "--once"]
    )
    assert result.returncode == EXIT_DIED_HARD  # fails closed
    assert read_status(status)["state"] == "died_hard"
    assert "different pid" in result.stderr


def test_missing_log_file_with_a_dead_process_is_still_the_alert(paths, dead_pid: int) -> None:
    log, status = paths  # never created
    result = run_monitor(
        ["--log", str(log), "--status-file", str(status), "--pid", str(dead_pid), "--once"]
    )
    assert result.returncode == EXIT_DIED_HARD
    assert read_status(status)["log_age_seconds"] is None


# --- status file integrity --------------------------------------------------
def test_status_file_is_always_valid_json_and_carries_its_schema(paths) -> None:
    log, status = paths
    log.write_text("running\n", encoding="utf-8")
    run_monitor(
        ["--log", str(log), "--status-file", str(status), "--pid", str(os.getpid()), "--once"]
    )
    record = read_status(status)
    assert record["schema"] == "probe_liveness/v1"
    assert record["checked_at_utc"].endswith("Z")
    assert not list(status.parent.glob("status.json.tmp.*"))  # atomic write left no debris


def test_status_file_is_overwritten_not_appended(paths) -> None:
    log, status = paths
    log.write_text("running\n", encoding="utf-8")
    args = ["--log", str(log), "--status-file", str(status), "--pid", str(os.getpid()), "--once"]
    run_monitor(args)
    run_monitor(args)
    read_status(status)  # would raise if the second write appended


# --- the two scripts must agree on the footer strings -----------------------
def test_footer_markers_match_the_launcher_that_writes_them() -> None:
    """The monitor greps for text the launcher prints. Nothing but this test
    stops one from being reworded without the other — and the failure mode is
    silent: every completed run would be reported as a hard death."""
    launcher = (REPO_ROOT / "scripts" / "launch_probe.sh").read_text(encoding="utf-8")
    monitor = SCRIPT.read_text(encoding="utf-8")

    assert 'FOOTER_OK="RUN COMPLETE"' in monitor
    assert 'FOOTER_FAIL="RUN DID NOT COMPLETE"' in monitor
    assert 'EXIT_RECORD_PREFIX="PROBE_EXIT_RECORD"' in monitor

    assert 'echo "RUN COMPLETE' in launcher
    assert 'echo "RUN DID NOT COMPLETE' in launcher
    assert "PROBE_EXIT_RECORD pid=$$ exit=${status}" in launcher


def test_launcher_exit_record_survives_a_dry_run_check() -> None:
    """The record line must be inside the EXIT trap, so it is written on every
    path out of the script — including the failure paths, which are the ones
    that matter."""
    launcher = (REPO_ROOT / "scripts" / "launch_probe.sh").read_text(encoding="utf-8")
    trap_body = launcher.split("on_exit() {", 1)[1].split("\ntrap on_exit EXIT", 1)[0]
    assert "PROBE_EXIT_RECORD" in trap_body
