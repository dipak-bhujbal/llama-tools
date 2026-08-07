"""Cumulative, append-only record of billable time for one stage.

A per-invocation timer is not a cap. The first backend gave every resume a fresh
full allowance, so a stage that crashed four times could bill four times the
approved ceiling while every individual run looked compliant. The allowance has
to be a property of the *stage*, not of the process.

This module is that property: an append-only JSONL beside the mining ledger,
recording one `session_start` and one `session_end` per invocation. Consumed
time is the sum of completed sessions plus the live one, so the remaining
allowance shrinks across resumes and reaches zero exactly once.

Two deliberate choices:

- **A session with no end still counts.** A process killed without writing
  `session_end` was billed for the time it held the pod. Its cost is charged
  from `started_at` to the successor's start, which over-counts idle gaps rather
  than under-counting billed time. Erring the other way would let a hard kill
  launder spend.
- **Wall-clock timestamps, not a monotonic clock.** Monotonic clocks do not
  survive a process boundary, and the thing being bounded here — a provider
  billing by the second — is wall-clock by nature.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPEND_LEDGER_NAME = "spend_ledger.jsonl"


class SpendLedgerError(RuntimeError):
    """The stage's spend history cannot be trusted, so nothing may be billed."""


@dataclass(frozen=True)
class StageAllowance:
    """What remains of the approved ceiling for this stage."""

    total_seconds: int
    consumed_seconds: float
    sessions: int

    @property
    def remaining_seconds(self) -> float:
        return self.total_seconds - self.consumed_seconds

    @property
    def exhausted(self) -> bool:
        return self.remaining_seconds <= 0


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            # A truncated tail is the normal shape of a crash; anything else is
            # not something a spend record may be guessed through.
            if index == len(path.read_text().splitlines()) - 1:
                break
            raise SpendLedgerError(f"{path}: unparseable spend record at line {index+1}") from exc
    return records


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def consumed_seconds(out_dir: Path, now: float) -> tuple[float, int]:
    """Billable seconds already spent on this stage, and the session count."""
    records = _read(Path(out_dir) / SPEND_LEDGER_NAME)
    starts = [r for r in records if r.get("event") == "session_start"]
    ends = {r.get("session"): r for r in records if r.get("event") == "session_end"}

    total = 0.0
    for position, start in enumerate(starts):
        end = ends.get(start.get("session"))
        if end is not None:
            total += max(0.0, float(end["at"]) - float(start["at"]))
        elif position + 1 < len(starts):
            # Killed without an end record: charge up to the next start rather
            # than forgiving it. A hard kill must not be cheaper than a clean one.
            total += max(0.0, float(starts[position + 1]["at"]) - float(start["at"]))
        else:
            total += max(0.0, now - float(start["at"]))
    return total, len(starts)


def open_session(
    out_dir: Path,
    total_seconds: int,
    rate_usd: float,
    cap_usd: float,
    now: float,
) -> tuple[str, StageAllowance]:
    """Record the start of a billable session and return what is left.

    Raises before writing anything if the stage's allowance is already spent, so
    an exhausted stage cannot be restarted into a fresh window.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    already, sessions = consumed_seconds(out_dir, now)
    allowance = StageAllowance(total_seconds, already, sessions)
    if allowance.exhausted:
        raise SpendLedgerError(
            f"this stage has already consumed {already:.0f}s of its "
            f"{total_seconds}s approved allowance across {sessions} session(s); "
            f"a resume does not grant a new one. Raising the ceiling is an owner "
            f"decision, not a restart"
        )

    session = f"s{sessions + 1:04d}"
    _append(
        out_dir / SPEND_LEDGER_NAME,
        {
            "event": "session_start",
            "session": session,
            "at": now,
            "rate_usd_per_hour": rate_usd,
            "cap_usd": cap_usd,
            "total_allowance_seconds": total_seconds,
            "consumed_before_seconds": already,
        },
    )
    return session, allowance


def close_session(
    out_dir: Path, session: str, now: float, exit_reason: str, rate_usd: float
) -> dict[str, Any]:
    """Record the end of a billable session and the cost it implies."""
    out_dir = Path(out_dir)
    consumed, sessions = consumed_seconds(out_dir, now)
    record = {
        "event": "session_end",
        "session": session,
        "at": now,
        "exit_reason": exit_reason,
        "stage_consumed_seconds": consumed,
        "stage_estimated_cost_usd": round(consumed / 3600 * rate_usd, 4),
        "sessions": sessions,
    }
    _append(out_dir / SPEND_LEDGER_NAME, record)
    return record
