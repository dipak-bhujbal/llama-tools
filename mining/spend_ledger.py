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
    """What remains of the approved ceiling, denominated in dollars.

    Seconds were the wrong unit. A stage that burns 6,000s at $0.53/hr and
    resumes at $0.33/hr has spent $0.883 of a $0.92 compute budget, but a
    seconds-based allowance would hand it another 1,200s — $0.11 more — pushing
    the total past the ceiling and into the storage reserve. Money is what the
    owner approved, so money is what is counted.
    """

    budget_usd: float
    consumed_usd: float
    sessions: int
    rate_usd_per_hour: float

    @property
    def remaining_usd(self) -> float:
        return self.budget_usd - self.consumed_usd

    @property
    def remaining_seconds(self) -> float:
        """What the remaining dollars buy at *this session's* rate."""
        if self.rate_usd_per_hour <= 0:
            return 0.0
        return max(0.0, self.remaining_usd / self.rate_usd_per_hour * 3600)

    @property
    def exhausted(self) -> bool:
        return self.remaining_usd <= 0


REQUIRED_START_FIELDS = (
    "session", "at", "rate_usd_per_hour", "cap_usd", "budget_usd",
)


def _read(path: Path) -> list[dict[str, Any]]:
    """Parse the whole chain, or refuse.

    A truncated final line is the normal shape of a crash, and the first version
    of this silently skipped it — then the next append concatenated onto the
    fragment, so the new session became invisible and the stage's consumed total
    reset to zero. A spend record is not something to recover heuristically:
    a damaged tail refuses until a human looks at it.
    """
    if not path.exists():
        return []
    payload = path.read_text()
    if payload and not payload.endswith("\n"):
        raise SpendLedgerError(
            f"{path} ends mid-record, so a previous session was killed while "
            f"writing. Its cost cannot be read and must not be guessed: inspect "
            f"the file and repair it deliberately before billing more time"
        )

    records: list[dict[str, Any]] = []
    for index, line in enumerate(payload.splitlines()):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpendLedgerError(
                f"{path}: unparseable spend record at line {index + 1}"
            ) from exc
        if record.get("event") == "session_start":
            missing = [f for f in REQUIRED_START_FIELDS if f not in record]
            if missing:
                raise SpendLedgerError(
                    f"{path}: session_start at line {index + 1} is missing {missing}"
                )
        records.append(record)
    return records


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def consumed_usd(out_dir: Path, now: float) -> tuple[float, int, float]:
    """Dollars already spent on this stage, the session count, and the seconds.

    Each session is priced at **its own recorded rate**. Pricing the whole
    history at the latest rate is what let a resume at a cheaper rate report
    $0.66 for time that actually cost $0.99.
    """
    records = _read(Path(out_dir) / SPEND_LEDGER_NAME)
    starts = [r for r in records if r.get("event") == "session_start"]
    ends = {r.get("session"): r for r in records if r.get("event") == "session_end"}

    total_usd = 0.0
    total_seconds = 0.0
    for position, start in enumerate(starts):
        end = ends.get(start.get("session"))
        if end is not None:
            seconds = max(0.0, float(end["at"]) - float(start["at"]))
        elif position + 1 < len(starts):
            # Killed without an end record: charge up to the next start rather
            # than forgiving it. A hard kill must not be cheaper than a clean one.
            seconds = max(0.0, float(starts[position + 1]["at"]) - float(start["at"]))
        else:
            seconds = max(0.0, now - float(start["at"]))
        total_seconds += seconds
        total_usd += seconds / 3600 * float(start["rate_usd_per_hour"])
    return total_usd, len(starts), total_seconds


def open_session(
    out_dir: Path,
    budget_usd: float,
    rate_usd: float,
    cap_usd: float,
    now: float,
) -> tuple[str, StageAllowance]:
    """Record the start of a billable session and return what is left.

    Raises before writing anything if the allowance is spent or if this session
    disagrees with the stage's established identity, so neither a restart nor a
    changed rate can widen a ceiling the owner set once.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SPEND_LEDGER_NAME

    records = _read(path)
    starts = [r for r in records if r.get("event") == "session_start"]
    if starts:
        first = starts[0]
        drift = {
            key: (first.get(key), value)
            for key, value in (("cap_usd", cap_usd), ("budget_usd", budget_usd))
            if first.get(key) != value
        }
        if drift:
            raise SpendLedgerError(
                f"this stage was opened under {dict((k, v[0]) for k, v in drift.items())} "
                f"and is now being resumed under {dict((k, v[1]) for k, v in drift.items())}. "
                f"A resume may not restate the ceiling it is spending against"
            )

    already_usd, sessions, already_seconds = consumed_usd(out_dir, now)
    allowance = StageAllowance(budget_usd, already_usd, sessions, rate_usd)
    if allowance.exhausted:
        raise SpendLedgerError(
            f"this stage has already spent ${already_usd:.4f} of its "
            f"${budget_usd:.2f} approved compute budget across {sessions} "
            f"session(s) ({already_seconds:.0f}s); a resume does not grant a new "
            f"one. Raising the ceiling is an owner decision, not a restart"
        )

    session = f"s{sessions + 1:04d}"
    _append(
        path,
        {
            "event": "session_start",
            "session": session,
            "at": now,
            "rate_usd_per_hour": rate_usd,
            "cap_usd": cap_usd,
            "budget_usd": budget_usd,
            "consumed_before_usd": already_usd,
            "consumed_before_seconds": already_seconds,
        },
    )
    _fsync_dir(out_dir)
    return session, allowance


def close_session(
    out_dir: Path, session: str, now: float, exit_reason: str, rate_usd: float
) -> dict[str, Any]:
    """Record the end of a billable session and the cost it implies."""
    out_dir = Path(out_dir)
    consumed, sessions, seconds = consumed_usd(out_dir, now)
    record = {
        "event": "session_end",
        "session": session,
        "at": now,
        "exit_reason": exit_reason,
        "stage_consumed_seconds": seconds,
        # Each session priced at its own rate, summed. Never the whole history
        # at the latest rate — that under-reports a resume onto a cheaper pod.
        "stage_cost_usd": round(consumed, 4),
        "sessions": sessions,
    }
    _append(out_dir / SPEND_LEDGER_NAME, record)
    return record
