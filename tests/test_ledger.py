"""Tests for the append-only mining ledger.

These exercise the guarantees the ledger exists to provide: durable
append-only writes, prompt dedup across resumed runs, rollback via
tombstones rather than in-place edits, and tolerance of a crash-truncated
final line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mining.ledger import Ledger


def test_append_then_processed_ids_round_trip(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append({"prompt_id": "p1", "text": "hello"})
    ledger.append({"prompt_id": "p2", "text": "world"})

    assert ledger.processed_ids() == {"p1", "p2"}


def test_duplicate_prompt_id_is_rejected(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append({"prompt_id": "p1"})

    with pytest.raises(ValueError, match="p1"):
        ledger.append({"prompt_id": "p1"})


def test_missing_prompt_id_raises(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")

    with pytest.raises(ValueError, match="prompt_id"):
        ledger.append({"text": "no id here"})


def test_redo_last_supersedes_exactly_the_last_n(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append({"prompt_id": "p1"})
    ledger.append({"prompt_id": "p2"})
    ledger.append({"prompt_id": "p3"})

    superseded_count = ledger.redo_last(2)

    assert superseded_count == 2
    assert ledger.processed_ids() == {"p1"}


def test_rolled_back_prompt_can_be_appended_again(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append({"prompt_id": "p1"})
    ledger.redo_last(1)

    assert "p1" not in ledger.processed_ids()

    # The whole point of rollback: this must not raise a dedup error.
    ledger.append({"prompt_id": "p1", "attempt": 2})

    assert ledger.processed_ids() == {"p1"}


def test_redo_last_never_rewrites_existing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    ledger.append({"prompt_id": "p2"})

    before = path.read_bytes()
    ledger.redo_last(1)
    after = path.read_bytes()

    # Append-only guarantee: every byte written before the rollback is
    # still there, unchanged, at the start of the file. redo_last only
    # ever adds a new tombstone line after it.
    assert after.startswith(before)
    assert len(after) > len(before)


def test_redo_last_beyond_available_records_is_not_an_error(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append({"prompt_id": "p1"})

    superseded_count = ledger.redo_last(5)

    assert superseded_count == 1
    assert ledger.processed_ids() == set()


def test_truncated_final_line_is_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    ledger.append({"prompt_id": "p2"})

    # Simulate a crash mid-write: chop the last line off before its
    # closing brace and newline, leaving invalid trailing JSON.
    raw = path.read_bytes()
    last_newline = raw.rindex(b"\n", 0, len(raw) - 1)
    truncated = raw[: last_newline + 1] + b'{"prompt_id": "p2", "se'
    path.write_bytes(truncated)

    reopened = Ledger(path)

    assert reopened.processed_ids() == {"p1"}
    summary = reopened.summary()
    assert summary["truncated_tail"] is True
    assert summary["active"] == 1
    # total_lines counts what is physically on disk (including the torn
    # fragment); parsed_records counts what was readable. The gap between
    # them is the evidence that something was lost.
    assert summary["total_lines"] == 2
    assert summary["parsed_records"] == 1


def test_seq_is_monotonic_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = Ledger(path)
    first.append({"prompt_id": "p1"})
    first.append({"prompt_id": "p2"})

    second = Ledger(path)
    second.append({"prompt_id": "p3"})

    lines = path.read_text().splitlines()
    seqs = [json.loads(line)["seq"] for line in lines]

    assert seqs == [1, 2, 3]


def test_summary_counts_active_superseded_and_tombstones(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append({"prompt_id": "p1"})
    ledger.append({"prompt_id": "p2"})
    ledger.append({"prompt_id": "p3"})
    ledger.redo_last(2)

    summary = ledger.summary()

    assert summary["active"] == 1
    assert summary["superseded"] == 2
    assert summary["tombstones"] == 2
    assert summary["total_lines"] == 5
    assert summary["truncated_tail"] is False


def test_corrupt_line_that_is_not_last_raises(tmp_path):
    """A corrupt line mid-file means something edited the ledger outside
    append()/redo_last(), which is exactly the invariant this file exists to
    protect. That is not the survivable crash case, so it must fail loudly."""
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    ledger.append({"prompt_id": "p2"})
    ledger.append({"prompt_id": "p3"})

    lines = path.read_bytes().split(b"\n")
    lines[1] = b'{"prompt_id": "p2", "seq":'  # corrupt a NON-final line
    path.write_bytes(b"\n".join(lines))

    with pytest.raises(ValueError, match="corrupt ledger line"):
        Ledger(path).processed_ids()


def test_summary_distinguishes_physical_lines_from_parsed_records(tmp_path):
    """total_lines counts what is on disk; parsed_records counts what read.
    A truncated tail must show up as a gap between them, so a reader can tell
    from the summary alone that something was unreadable."""
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    ledger.append({"prompt_id": "p2"})

    clean = ledger.summary()
    assert clean["total_lines"] == clean["parsed_records"] == 2
    assert clean["truncated_tail"] is False

    with path.open("a") as f:  # simulate a crash mid-write
        f.write('{"prompt_id": "p3", "se')

    torn = Ledger(path).summary()
    assert torn["truncated_tail"] is True
    assert torn["total_lines"] == 3
    assert torn["parsed_records"] == 2
    assert torn["active"] == 2
