"""Tests for the append-only mining ledger.

These exercise the guarantees the ledger exists to provide: durable
append-only writes, prompt dedup across resumed runs, rollback via
tombstones rather than in-place edits, and tolerance of a crash-truncated
final line.
"""

from __future__ import annotations

import json
import os
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


# --------------------------------------------------------------------------
# Append-after-truncated-tail: rotation to a hash-linked segment.
#
# `_read_records` tolerates a crash-truncated final line, but tolerating it on
# read is not enough. The next append opens the same file in "a" mode and
# writes after the fragment, splicing fragment and new record into one line.
# That line is then a corrupt line that is NOT last, which is the one condition
# the ledger treats as a hard integrity failure — so a single crash mid-write
# would make the ledger permanently unreadable and silently destroy the first
# record of the resumed run. Rotating to a new hash-linked segment fixes this
# without editing a single byte of the damaged one.
# --------------------------------------------------------------------------


def _torn_tail(path: Path, fragment: bytes = b'{"prompt_id": "p3", "se') -> None:
    """Simulate a crash partway through writing the next record."""
    raw = path.read_bytes()
    assert raw.endswith(b"\n"), "fixture expects a clean file to damage"
    path.write_bytes(raw + fragment)


def test_append_after_truncated_tail_does_not_corrupt_the_ledger(tmp_path: Path) -> None:
    """The regression: appending after a torn tail must stay readable."""
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    ledger.append({"prompt_id": "p2"})
    _torn_tail(path)

    ledger.append({"prompt_id": "p3"})

    # Before the fix this raised ValueError("corrupt ledger line ...").
    assert ledger.processed_ids() == {"p1", "p2", "p3"}


def test_append_after_truncated_tail_preserves_the_damaged_bytes(tmp_path: Path) -> None:
    """Never silently truncate: the fragment is evidence and must survive."""
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    _torn_tail(path)
    damaged = path.read_bytes()

    ledger.append({"prompt_id": "p2"})

    assert path.read_bytes() == damaged, "damaged segment was modified"


def test_append_after_truncated_tail_rotates_to_a_new_segment(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    _torn_tail(path)
    ledger.append({"prompt_id": "p2"})

    segments = ledger.segment_paths()
    assert len(segments) == 2
    assert segments[0] == path
    assert segments[1].exists()


def test_rotated_segment_is_hash_linked_to_its_predecessor(tmp_path: Path) -> None:
    """The chain is verifiable: the header pins the previous segment's bytes."""
    import hashlib

    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    _torn_tail(path)
    ledger.append({"prompt_id": "p2"})

    header = json.loads(ledger.segment_paths()[1].read_text().splitlines()[0])
    assert header["type"] == "segment_open"
    assert header["prev_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert header["prev_bytes"] == len(path.read_bytes())


def test_broken_segment_chain_fails_closed(tmp_path: Path) -> None:
    """A tampered predecessor must be detected, not read past."""
    from mining.ledger import LedgerIntegrityError

    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    _torn_tail(path)
    ledger.append({"prompt_id": "p2"})

    # Rewrite history in the sealed segment: same line count, different bytes.
    sealed = path.read_bytes().replace(b'"p1"', b'"pX"')
    path.write_bytes(sealed)

    with pytest.raises(LedgerIntegrityError):
        ledger.processed_ids()


def test_seq_stays_monotonic_across_a_rotation(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    ledger.append({"prompt_id": "p2"})
    _torn_tail(path)
    ledger.append({"prompt_id": "p3"})
    ledger.append({"prompt_id": "p4"})

    seqs = [rec["seq"] for rec in ledger.records()]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))


def test_dedup_and_rollback_still_span_segments(tmp_path: Path) -> None:
    """Rotation must not create a second, independent namespace."""
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    _torn_tail(path)
    ledger.append({"prompt_id": "p2"})

    with pytest.raises(ValueError):
        ledger.append({"prompt_id": "p1"})

    assert ledger.redo_last(2) == 2
    assert ledger.processed_ids() == set()


def test_summary_reports_segments_and_the_surviving_torn_tail(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    _torn_tail(path)
    ledger.append({"prompt_id": "p2"})

    summary = ledger.summary()
    assert summary["segments"] == 2
    assert summary["truncated_tail"] is True, "the fragment is still on disk"
    assert summary["active"] == 2
    assert summary["total_lines"] - summary["parsed_records"] == 1


# --------------------------------------------------------------------------
# Review cycle 1 (msg 2020): three integrity holes left by the first segment
# implementation. Each of these failed before the fix.
# --------------------------------------------------------------------------


def test_redo_last_does_not_duplicate_seq_across_a_rotation(tmp_path: Path) -> None:
    """seq must be allocated after rotation, for tombstones too, not just appends.

    The first implementation fixed the ordering in append() but left redo_last()
    computing seq before _append_raw() rotated, so the segment header and the
    tombstone both took the same number — recreating the exact ambiguity the
    segment design exists to remove.
    """
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    _torn_tail(path)

    ledger.redo_last(1)

    seqs = [rec["seq"] for rec in ledger.records()]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs)), f"duplicate seq: {seqs}"
    assert ledger.processed_ids() == set()


def test_non_increasing_seq_fails_closed(tmp_path: Path) -> None:
    """The read-time invariant, independent of how the duplicate got there."""
    from mining.ledger import LedgerIntegrityError

    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    ledger.append({"prompt_id": "p2"})
    path.write_bytes(path.read_bytes() + json.dumps({"prompt_id": "p3", "seq": 2}).encode() + b"\n")

    with pytest.raises(LedgerIntegrityError, match="strictly increasing"):
        ledger.processed_ids()


def _rotate_once(ledger: Ledger, path: Path, prompt_id: str) -> None:
    _torn_tail(path, b'{"partial": "wr')
    ledger.append({"prompt_id": prompt_id})


def test_missing_middle_segment_fails_closed(tmp_path: Path) -> None:
    """A gap must raise, not silently truncate history to the clean prefix.

    Counting upward from zero until a file is missing reports a short,
    well-formed-looking ledger and loses every later segment without a word.
    """
    from mining.ledger import LedgerIntegrityError

    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    _rotate_once(ledger, path, "p2")
    _rotate_once(ledger, ledger.segment_paths()[1], "p3")
    assert len(ledger.segment_paths()) == 3

    ledger.segment_paths()[1].unlink()

    with pytest.raises(LedgerIntegrityError, match="not contiguous"):
        ledger.segment_paths()
    with pytest.raises(LedgerIntegrityError):
        ledger.processed_ids()


def test_malformed_segment_name_fails_closed(tmp_path: Path) -> None:
    from mining.ledger import LedgerIntegrityError

    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    (tmp_path / "ledger.segXX.jsonl").write_text("{}\n")

    with pytest.raises(LedgerIntegrityError, match="malformed"):
        ledger.segment_paths()


def test_rotation_publication_is_atomic(tmp_path: Path, monkeypatch) -> None:
    """A crash during rotation must leave no successor, not a partial one.

    The first implementation touched the new path and then appended the header.
    A crash in between published an empty segment, and every subsequent read
    raised forever — unrecoverable, because the retry could not get past it.
    """
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    _torn_tail(path)

    real_link = os.link

    def crash_before_publish(src, dst):
        raise RuntimeError("crash before publication")

    monkeypatch.setattr(os, "link", crash_before_publish)
    with pytest.raises(RuntimeError):
        ledger.append({"prompt_id": "p2"})

    # Nothing partial was left behind, and no temp file leaked.
    assert not (tmp_path / "ledger.seg001.jsonl").exists()
    assert list(tmp_path.glob(".*tmp")) == []

    # The retry succeeds: the ledger was never wedged.
    monkeypatch.setattr(os, "link", real_link)
    ledger.append({"prompt_id": "p2"})
    assert ledger.processed_ids() == {"p1", "p2"}


def test_empty_successor_segment_fails_closed(tmp_path: Path) -> None:
    """Defence in depth: a segment left empty by an older build must not be read past."""
    from mining.ledger import LedgerIntegrityError

    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    _torn_tail(path)
    (tmp_path / "ledger.seg001.jsonl").touch()

    with pytest.raises(LedgerIntegrityError, match="missing its segment_open header"):
        ledger.processed_ids()


def test_segment_header_must_name_the_right_predecessor(tmp_path: Path) -> None:
    from mining.ledger import LedgerIntegrityError

    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    _rotate_once(ledger, path, "p2")

    seg1 = ledger.segment_paths()[1]
    lines = seg1.read_text().splitlines()
    header = json.loads(lines[0])
    header["prev_segment"] = "somebody_elses_ledger.jsonl"
    seg1.write_text("\n".join([json.dumps(header, sort_keys=True), *lines[1:]]) + "\n")

    with pytest.raises(LedgerIntegrityError, match="names predecessor"):
        ledger.processed_ids()


def test_segment_header_must_be_first_and_only(tmp_path: Path) -> None:
    from mining.ledger import LedgerIntegrityError

    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    _rotate_once(ledger, path, "p2")

    seg1 = ledger.segment_paths()[1]
    header = json.loads(seg1.read_text().splitlines()[0])
    seg1.write_text(seg1.read_text() + json.dumps(header, sort_keys=True) + "\n")

    with pytest.raises(LedgerIntegrityError, match="exactly once, first"):
        ledger.processed_ids()


def test_segment_header_index_must_match_its_filename(tmp_path: Path) -> None:
    from mining.ledger import LedgerIntegrityError

    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    _rotate_once(ledger, path, "p2")

    seg1 = ledger.segment_paths()[1]
    lines = seg1.read_text().splitlines()
    header = json.loads(lines[0])
    header["segment"] = 7
    seg1.write_text("\n".join([json.dumps(header, sort_keys=True), *lines[1:]]) + "\n")

    with pytest.raises(LedgerIntegrityError, match="claims segment 7"):
        ledger.processed_ids()


def test_repeated_rotations_stay_consistent(tmp_path: Path) -> None:
    """Three crashes in a row still produce one verifiable chain."""
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append({"prompt_id": "p1"})
    _rotate_once(ledger, path, "p2")
    _rotate_once(ledger, ledger.segment_paths()[1], "p3")
    _rotate_once(ledger, ledger.segment_paths()[2], "p4")

    assert len(ledger.segment_paths()) == 4
    assert ledger.processed_ids() == {"p1", "p2", "p3", "p4"}
    seqs = [rec["seq"] for rec in ledger.records()]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))
    assert ledger.summary()["segments"] == 4
    assert ledger.redo_last(4) == 4
    assert ledger.processed_ids() == set()
