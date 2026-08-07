"""Tests for the external, seconds-only launch envelope."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def launcher_module():
    path = Path(__file__).parents[1] / "scripts" / "launch_mining_stage.py"
    spec = importlib.util.spec_from_file_location("launch_mining_stage", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_session_is_counted_once(tmp_path) -> None:
    module = launcher_module()
    ledger = tmp_path / "sessions.jsonl"
    ledger.write_text(
        json.dumps({"event": "session_start", "session": "s1", "started_at": 10.0})
        + "\n"
        + json.dumps({"event": "session_end", "session": "s1", "elapsed_seconds": 7.0})
        + "\n"
    )
    assert module.elapsed_before(ledger, now=100.0) == 7.0


def test_live_session_is_counted_to_now(tmp_path) -> None:
    module = launcher_module()
    ledger = tmp_path / "sessions.jsonl"
    ledger.write_text(
        json.dumps({"event": "session_start", "session": "s1", "started_at": 10.0})
        + "\n"
    )
    assert module.elapsed_before(ledger, now=17.5) == 7.5
