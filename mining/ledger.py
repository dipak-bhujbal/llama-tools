"""Append-only ledger for the mining job's per-prompt sampling records.

The mining job is long-running and samples model generations for potentially
tens of thousands of prompts. It must survive crashes (OOM, preemption, a bad
generation that hangs the process) and resume without re-doing work that was
already paid for in compute. Its output also becomes evidence cited in a
research paper: a reviewer must be able to trust that the record of "what was
sampled, when, and what was later invalidated" was never quietly rewritten
after the fact.

Both of those needs point to the same design: **the ledger is append-only and
is never hand-edited.**

- Crash safety: because every record is a single flushed-and-fsynced JSONL
  line, a crash can only ever produce a clean prefix of complete lines plus,
  at worst, one incomplete trailing line. Resuming just means re-reading the
  file and skipping that trailing fragment — there is no in-place state to
  corrupt.
- Evidentiary integrity: mistakes happen (a batch sampled with a bad prompt
  template, a run that needs to be redone). Instead of deleting or editing
  the offending lines, we append tombstone ("redo") records that supersede
  them. The original lines stay on disk, byte-for-byte, forever. Anyone
  auditing the ledger later can see both what happened and that it was later
  corrected — the correction is itself part of the permanent record.

## Segments

Tolerating a truncated final line on *read* is not sufficient. Appending after
one splices the fragment and the new record onto a single physical line, which
loses the new record, duplicates a `seq`, and finally makes the file raise on
every subsequent read. So a ledger is a *chain of segments* rather than one
file: `ledger.jsonl`, then `ledger.seg001.jsonl`, and so on.

When a write is about to happen and the active segment ends in a fragment, the
damaged segment is **sealed, never repaired** — its bytes, fragment included,
stay exactly as the crash left them — and writing continues in a fresh segment
whose first line records the sealed segment's sha256 and length. That makes the
chain verifiable end to end: a sealed segment that changes by even one byte is
detected on the next read and fails closed rather than being read past.

`seq`, prompt dedup, and tombstones all span the whole chain, so rotation is
invisible to callers apart from `segment_paths()` and `summary()["segments"]`.

This module does not interpret the contents of a record beyond `prompt_id`
(required, used for dedup/resume) and the reserved `type` values `"redo"`
(tombstones) and `"segment_open"` (segment headers). Everything else in a
record is the caller's business.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

# Record types the ledger itself owns. They carry no `prompt_id` and are never
# counted as units of work.
CONTROL_TYPES = frozenset({"redo", "segment_open"})

_SEGMENT_RE = re.compile(r"\.seg(\d{3,})$")


class LedgerIntegrityError(ValueError):
    """A sealed segment's bytes on disk do not match the hash that follows it."""


class Ledger:
    """An append-only JSONL ledger of per-prompt mining records.

    Each non-control record represents one unit of completed work (e.g.
    "sampled generations for this prompt"), keyed by `prompt_id`. Tombstone
    records (`type: "redo"`) mark earlier records as superseded without
    touching the bytes already written for them.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    # ------------------------------------------------------------ segments --

    def _segment_path(self, index: int) -> Path:
        """Path of segment `index`; segment 0 is the base path itself."""
        if index == 0:
            return self.path
        stem = self.path.name[: -len(self.path.suffix)] if self.path.suffix else self.path.name
        return self.path.with_name(f"{stem}.seg{index:03d}{self.path.suffix}")

    def segment_paths(self) -> list[Path]:
        """Existing segments in chain order, starting with the base path."""
        segments = [self.path]
        index = 1
        while True:
            candidate = self._segment_path(index)
            if not candidate.exists():
                break
            segments.append(candidate)
            index += 1
        return segments

    @staticmethod
    def _parse_segment(path: Path) -> tuple[list[dict[str, Any]], bool]:
        """Parse one segment. Returns (records, ends_in_a_fragment)."""
        raw = path.read_bytes()
        if not raw:
            return [], False

        lines = raw.split(b"\n")
        # A well-formed file ends with a trailing newline, so the split
        # produces one empty element at the end; drop it before checking
        # for truncation.
        if lines and lines[-1] == b"":
            lines.pop()

        records: list[dict[str, Any]] = []
        truncated_tail = False
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    # Incomplete final line: a crash happened after the
                    # write started but before flush()+fsync() completed.
                    # It is left on disk untouched and the segment is sealed.
                    truncated_tail = True
                else:
                    # A corrupt line that isn't the last one is a real
                    # integrity problem, not an in-flight write — the
                    # append-only guarantee (flush+fsync per line, never
                    # edited afterward) means this should never happen.
                    raise ValueError(f"corrupt ledger line {i} in {path}") from None
        return records, truncated_tail

    def _verify_link(self, sealed: Path, successor_records: list[dict[str, Any]]) -> None:
        """Check a sealed segment against the hash recorded by its successor."""
        header = successor_records[0] if successor_records else None
        if header is None or header.get("type") != "segment_open":
            raise LedgerIntegrityError(
                f"segment {self._segment_index(sealed) + 1} is missing its segment_open header"
            )

        raw = sealed.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if header.get("prev_sha256") != actual:
            raise LedgerIntegrityError(
                f"sealed segment {sealed.name} was modified after it was sealed: "
                f"expected sha256 {header.get('prev_sha256')}, found {actual}"
            )
        if header.get("prev_bytes") != len(raw):
            raise LedgerIntegrityError(
                f"sealed segment {sealed.name} changed length after it was sealed: "
                f"expected {header.get('prev_bytes')} bytes, found {len(raw)}"
            )

    @staticmethod
    def _segment_index(path: Path) -> int:
        match = _SEGMENT_RE.search(path.name[: -len(path.suffix)] if path.suffix else path.name)
        return int(match.group(1)) if match else 0

    # ---------------------------------------------------------------- read --

    def _read_records(self) -> tuple[list[dict[str, Any]], bool]:
        """Parse every complete line across the whole segment chain.

        Returns the parsed records in chain order and whether any segment ends
        in a truncated fragment. Fragments reflect work that was never durably
        recorded and are simply redone by the caller.

        Raises `LedgerIntegrityError` if a sealed segment's bytes no longer
        match the hash its successor recorded — the ledger fails closed rather
        than reporting a resumable state it cannot vouch for.
        """
        segments = self.segment_paths()
        parsed = [self._parse_segment(segment) for segment in segments]

        for i, sealed in enumerate(segments[:-1]):
            self._verify_link(sealed, parsed[i + 1][0])

        records: list[dict[str, Any]] = []
        truncated_tail = False
        for segment_records, segment_truncated in parsed:
            records.extend(segment_records)
            truncated_tail = truncated_tail or segment_truncated
        return records, truncated_tail

    def records(self) -> list[dict[str, Any]]:
        """Every parsed record across the chain, in write order."""
        return self._read_records()[0]

    def _effective_state(self) -> tuple[list[dict[str, Any]], set[int], bool]:
        """Return (all records, seqs superseded by a redo, truncated_tail)."""
        records, truncated_tail = self._read_records()
        superseded_seqs: set[int] = set()
        for rec in records:
            if rec.get("type") == "redo":
                seq = rec.get("supersedes_seq")
                if seq is not None:
                    superseded_seqs.add(seq)
        return records, superseded_seqs, truncated_tail

    def _next_seq(self) -> int:
        records, _ = self._read_records()
        if not records:
            return 1
        return max(rec["seq"] for rec in records) + 1

    # --------------------------------------------------------------- write --

    def _write_line(self, path: Path, record: dict[str, Any]) -> None:
        """Serialize one record as a single JSONL line and durably write it.

        flush() pushes the line out of Python's buffer to the OS; fsync()
        forces the OS to write it to durable storage. Together they mean a
        crash immediately after this call returns cannot lose or partially
        write this line — the worst a crash *during* the write can do is
        leave this line truncated, which seals the segment.
        """
        line = json.dumps(record, sort_keys=True) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def _seal_and_rotate(self) -> Path:
        """Seal the damaged active segment and open its hash-linked successor.

        The damaged segment is not repaired, truncated, or reopened. Its bytes
        are hashed as they stand — fragment included — and that hash is written
        into the first line of the new segment, so the seal is verifiable by
        anyone auditing the chain later.
        """
        segments = self.segment_paths()
        sealed = segments[-1]
        raw = sealed.read_bytes()
        new_index = self._segment_index(sealed) + 1
        new_path = self._segment_path(new_index)

        header = {
            "type": "segment_open",
            "segment": new_index,
            "prev_segment": sealed.name,
            "prev_sha256": hashlib.sha256(raw).hexdigest(),
            "prev_bytes": len(raw),
            "reason": "truncated tail in previous segment (crash during write)",
            "seq": self._next_seq(),
        }
        new_path.touch(exist_ok=True)
        self._write_line(new_path, header)
        return new_path

    def _active_segment(self) -> Path:
        """The segment to append to, rotating first if the current one is torn.

        Every write path goes through here, so a fragment can never be
        appended after.
        """
        segments = self.segment_paths()
        active = segments[-1]
        _, truncated_tail = self._parse_segment(active)
        if truncated_tail:
            return self._seal_and_rotate()
        return active

    def _append_raw(self, record: dict[str, Any]) -> None:
        self._write_line(self._active_segment(), record)

    def append(self, record: dict[str, Any]) -> None:
        """Append one record, assigning it the next monotonic `seq`.

        Raises `ValueError` if `prompt_id` is missing, or if `prompt_id` is
        already active (appended and not since rolled back via
        `redo_last`) — this is the dedup guard that keeps a prompt from
        being double-counted across resumed runs.
        """
        if "prompt_id" not in record:
            raise ValueError("record is missing required field 'prompt_id'")

        prompt_id = record["prompt_id"]
        if prompt_id in self.processed_ids():
            raise ValueError(f"prompt_id {prompt_id!r} was already appended and not rolled back")

        # Rotate before computing `seq`, so the header cannot claim the same
        # sequence number as the record that follows it.
        target = self._active_segment()
        out = dict(record)
        out["seq"] = self._next_seq()
        self._write_line(target, out)

    def processed_ids(self) -> set[str]:
        """Return prompt_ids currently in effect (i.e. not superseded)."""
        records, superseded_seqs, _ = self._effective_state()
        ids: set[str] = set()
        for rec in records:
            if rec.get("type") in CONTROL_TYPES:
                continue
            if rec["seq"] in superseded_seqs:
                continue
            ids.add(rec["prompt_id"])
        return ids

    def redo_last(self, n: int) -> int:
        """Supersede the most recent `n` non-control, not-yet-superseded records.

        Implements rollback without rewriting history: each superseded
        record is marked by appending a new tombstone line
        (`{"type": "redo", "supersedes_seq": <seq>, ...}`); no existing
        line is ever modified or deleted. Rolling back more records than
        currently exist is not an error — it supersedes as many as it can
        and returns that count.
        """
        if n <= 0:
            return 0

        records, superseded_seqs, _ = self._effective_state()
        active = [
            rec
            for rec in records
            if rec.get("type") not in CONTROL_TYPES and rec["seq"] not in superseded_seqs
        ]
        active.sort(key=lambda rec: rec["seq"])
        to_supersede = active[-n:] if n <= len(active) else active

        for rec in to_supersede:
            tombstone = {
                "type": "redo",
                "supersedes_seq": rec["seq"],
                "prompt_id": rec["prompt_id"],
                "seq": self._next_seq(),
            }
            self._append_raw(tombstone)

        return len(to_supersede)

    def summary(self) -> dict[str, Any]:
        """Return counts describing the current state of the ledger.

        `total_lines` counts physical non-empty lines across every segment,
        including any truncated fragment. `parsed_records` counts lines that
        parsed. The two differ by exactly the surviving fragments, so a reader
        can tell from the summary alone whether anything on disk was
        unreadable — which matters when this file is cited as evidence.
        """
        records, truncated_tail = self._read_records()
        segments = self.segment_paths()
        physical_lines = sum(
            1
            for segment in segments
            for line in segment.read_bytes().split(b"\n")
            if line.strip()
        )
        superseded_seqs: set[int] = set()
        tombstones = 0
        for rec in records:
            if rec.get("type") == "redo":
                tombstones += 1
                seq = rec.get("supersedes_seq")
                if seq is not None:
                    superseded_seqs.add(seq)

        work = [rec for rec in records if rec.get("type") not in CONTROL_TYPES]
        superseded = sum(1 for rec in work if rec["seq"] in superseded_seqs)
        active = len(work) - superseded

        return {
            "total_lines": physical_lines,
            "parsed_records": len(records),
            "active": active,
            "superseded": superseded,
            "tombstones": tombstones,
            "truncated_tail": truncated_tail,
            "segments": len(segments),
        }
