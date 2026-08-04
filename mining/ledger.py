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

This module does not interpret the contents of a record beyond `prompt_id`
(required, used for dedup/resume) and the reserved `type: "redo"` marker used
for tombstones. Everything else in a record is the caller's business.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class Ledger:
    """An append-only JSONL ledger of per-prompt mining records.

    Each non-tombstone record represents one unit of completed work (e.g.
    "sampled generations for this prompt"), keyed by `prompt_id`. Tombstone
    records (`type: "redo"`) mark earlier records as superseded without
    touching the bytes already written for them.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _read_records(self) -> tuple[list[dict[str, Any]], bool]:
        """Parse every complete line in the ledger.

        Returns the parsed records in file order and whether the final line
        in the file was truncated (present but not valid JSON, e.g. because
        a crash happened mid-write). A truncated final line is skipped, not
        raised — it reflects work that was never durably recorded and is
        simply redone by the caller.
        """
        raw = self.path.read_bytes()
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
                    truncated_tail = True
                else:
                    # A corrupt line that isn't the last one is a real
                    # integrity problem, not an in-flight write — the
                    # append-only guarantee (flush+fsync per line, never
                    # edited afterward) means this should never happen.
                    raise ValueError(f"corrupt ledger line {i} in {self.path}") from None
        return records, truncated_tail

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

    def _append_raw(self, record: dict[str, Any]) -> None:
        """Serialize one record as a single JSONL line and durably write it.

        flush() pushes the line out of Python's buffer to the OS; fsync()
        forces the OS to write it to durable storage. Together they mean a
        crash immediately after this call returns cannot lose or partially
        write this line — the worst a crash *during* the write can do is
        leave this line truncated, which `_read_records` already tolerates.
        """
        line = json.dumps(record, sort_keys=True) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

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

        out = dict(record)
        out["seq"] = self._next_seq()
        self._append_raw(out)

    def processed_ids(self) -> set[str]:
        """Return prompt_ids currently in effect (i.e. not superseded)."""
        records, superseded_seqs, _ = self._effective_state()
        ids: set[str] = set()
        for rec in records:
            if rec.get("type") == "redo":
                continue
            if rec["seq"] in superseded_seqs:
                continue
            ids.add(rec["prompt_id"])
        return ids

    def redo_last(self, n: int) -> int:
        """Supersede the most recent `n` non-tombstone, not-yet-superseded records.

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
            if rec.get("type") != "redo" and rec["seq"] not in superseded_seqs
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

        `total_lines` counts physical non-empty lines on disk, including a
        truncated trailing fragment. `parsed_records` counts lines that
        parsed. The two differ by exactly the truncated tail, so a reader can
        tell from the summary alone whether anything on disk was unreadable —
        which matters when this file is cited as evidence.
        """
        records, truncated_tail = self._read_records()
        physical_lines = sum(
            1 for line in self.path.read_bytes().split(b"\n") if line.strip()
        )
        superseded_seqs: set[int] = set()
        tombstones = 0
        for rec in records:
            if rec.get("type") == "redo":
                tombstones += 1
                seq = rec.get("supersedes_seq")
                if seq is not None:
                    superseded_seqs.add(seq)

        non_tombstones = [rec for rec in records if rec.get("type") != "redo"]
        superseded = sum(1 for rec in non_tombstones if rec["seq"] in superseded_seqs)
        active = len(non_tombstones) - superseded

        return {
            "total_lines": physical_lines,
            "parsed_records": len(records),
            "active": active,
            "superseded": superseded,
            "tombstones": tombstones,
            "truncated_tail": truncated_tail,
        }
