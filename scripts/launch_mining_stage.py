#!/usr/bin/env python3
"""Launch one miner session under an external, persistent wall-clock envelope.

This wrapper owns provider lifecycle and elapsed-time accounting.  The miner
itself deliberately has no prices, rates, budgets, or billing receipts.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def elapsed_before(path: Path, now: float) -> float:
    if not path.exists():
        return 0.0
    starts: dict[str, float] = {}
    ended: dict[str, float] = {}
    total = 0.0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "session_end":
            ended[str(record["session"])] = float(record["elapsed_seconds"])
        elif record.get("event") == "session_start":
            starts[str(record["session"])] = float(record["started_at"])
    for session, started in starts.items():
        total += ended.get(session, max(0.0, now - started))
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "calibration"), default="pilot")
    parser.add_argument("--max-stage-seconds", type=int, required=True)
    parser.add_argument("--provider-terminate-seconds", type=int, required=True)
    parser.add_argument(
        "--attest-durable-root", action="store_true", required=True,
        help="explicitly attest that persistent-root survives provider termination",
    )
    parser.add_argument("--confirm-paid-run", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if not args.confirm_paid_run:
        parser.error("--confirm-paid-run is required for a model-backed launch")
    if args.max_stage_seconds <= 0 or args.provider_terminate_seconds <= 0:
        parser.error("stage and provider durations must be positive seconds")
    if args.provider_terminate_seconds > args.max_stage_seconds:
        parser.error("provider termination must not exceed the stage duration")

    root = args.persistent_root.resolve()
    out = args.out_dir.resolve()
    if not root.is_dir() or (root not in out.parents and out != root):
        parser.error("out-dir must be inside an existing persistent-root")
    out.mkdir(parents=True, exist_ok=True)
    timeout = shutil.which("timeout") or shutil.which("gtimeout")
    if timeout is None:
        parser.error("GNU timeout/gtimeout is required; refusing an uncapped launch")

    lock_path = root / ".mining-stage-launch.lock"
    ledger_path = root / "mining-stage-sessions.jsonl"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        now = time.time()
        try:
            consumed = elapsed_before(ledger_path, now)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"cannot trust the session ledger {ledger_path}: {exc}")
        remaining = args.max_stage_seconds - consumed
        if remaining <= 0:
            parser.error("stage elapsed-time allowance is exhausted")
        session = f"s{int(now * 1_000_000):016d}"
        token = secrets.token_hex(24)
        context_path = root / f"launch-context-{session}.json"
        context_path.write_text(json.dumps({
            "session": session, "token": token, "out_dir": str(out),
            "ledger": str(ledger_path),
            "max_stage_seconds": args.max_stage_seconds,
            "provider_terminate_seconds": args.provider_terminate_seconds,
        }, sort_keys=True) + "\n")
        with context_path.open("r+") as context_handle:
            context_handle.flush()
            os.fsync(context_handle.fileno())
        append_jsonl(ledger_path, {
            "event": "session_start", "session": session, "started_at": now,
            "max_stage_seconds": args.max_stage_seconds,
            "provider_terminate_seconds": args.provider_terminate_seconds,
        })
        command = [
            timeout, "--kill-after=30", str(int(remaining)), sys.executable,
            "-m", "mining.mine_pairs", "--stage", args.stage,
            "--out-dir", str(out), "--confirm-paid-run",
            "--persistent-root", str(root), "--attest-durable-root",
        ]
        if args.fresh:
            command.append("--fresh")
        started = time.monotonic()
        try:
            env = os.environ.copy()
            env["LLAMA_TOOLS_LAUNCH_CONTEXT"] = str(context_path)
            env["LLAMA_TOOLS_LAUNCH_TOKEN"] = token
            result = subprocess.run(
                command, check=False, cwd=Path(__file__).resolve().parents[1], env=env
            )
            returncode = result.returncode
        finally:
            elapsed = time.monotonic() - started
            append_jsonl(ledger_path, {
                "event": "session_end", "session": session,
                "ended_at": time.time(), "elapsed_seconds": elapsed,
                "returncode": returncode if "returncode" in locals() else None,
            })
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
