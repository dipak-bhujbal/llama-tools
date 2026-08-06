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
  at worst, one incomplete trailing line.
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
whose first line records the sealed segment's sha256 and length.

Three properties make that chain trustworthy rather than merely tidy:

- **One write path.** Every record — work, tombstone, or segment header —
  goes through `_commit()`, which rotates first, *then* allocates `seq`, then
  writes. A `seq` can therefore never be allocated against a pre-rotation view
  of the ledger and collide with the header that rotation just wrote.
- **Whole-chain enumeration.** Segments are discovered by globbing, not by
  counting upward from zero until a file is missing. A gap, a duplicate, or a
  malformed segment name fails closed; it never silently truncates history to
  the part that happens to be contiguous.
- **Atomic publication.** A new segment is built complete in a temp file,
  fsynced, and then linked into place only if nothing is there. A crash during
  rotation leaves either no successor or a complete one — never an empty or
  half-written file that wedges every future read.

Read-time verification then checks the chain end to end: sealed segments must
hash to what their successor recorded, headers must sit first in their segment
and name the right predecessor, `seq` must be a positive integer that strictly
increases across the whole chain, and every tombstone must name a real,
earlier, still-active record for the same prompt. Any violation raises
`LedgerIntegrityError` rather than returning a resumable-looking state the
ledger cannot vouch for.

The control records that carry those guarantees are the ledger's own: callers
append work, and only the ledger writes `redo` and `segment_open`. `append()`
refuses a caller-supplied control type, so no mining code can retire a record
from active work or forge a seal over history through the ordinary API.

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
import tempfile
from pathlib import Path
from typing import Any

# Record types the ledger itself owns. They carry no `prompt_id` and are never
# counted as units of work.
CONTROL_TYPES = frozenset({"redo", "segment_open"})

_SEGMENT_RE = re.compile(r"^(?P<stem>.+)\.seg(?P<index>\d{3,})$")


class LedgerIntegrityError(ValueError):
    """The ledger chain on disk does not match what it recorded about itself."""


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

    @property
    def _stem(self) -> str:
        name = self.path.name
        return name[: -len(self.path.suffix)] if self.path.suffix else name

    def _segment_path(self, index: int) -> Path:
        """Path of segment `index`; segment 0 is the base path itself."""
        if index == 0:
            return self.path
        return self.path.with_name(f"{self._stem}.seg{index:03d}{self.path.suffix}")

    def segment_paths(self) -> list[Path]:
        """Existing segments in chain order, starting with the base path.

        Enumerates every file matching the segment naming scheme rather than
        counting upward until one is missing. Counting upward silently drops
        all history after a gap — a deleted middle segment would make the
        ledger report a short, clean-looking prefix. Gaps, duplicate indices,
        and malformed names all fail closed instead.
        """
        found: dict[int, Path] = {}
        for candidate in self.path.parent.glob(f"{self._stem}.seg*{self.path.suffix}"):
            stem = (
                candidate.name[: -len(candidate.suffix)]
                if candidate.suffix
                else candidate.name
            )
            match = _SEGMENT_RE.match(stem)
            if match is None or match.group("stem") != self._stem:
                raise LedgerIntegrityError(
                    f"malformed ledger segment name: {candidate.name}"
                )
            index = int(match.group("index"))
            if index == 0:
                raise LedgerIntegrityError(
                    f"segment index 0 is reserved for {self.path.name}: {candidate.name}"
                )
            if index in found:
                raise LedgerIntegrityError(
                    f"duplicate ledger segment index {index}: "
                    f"{found[index].name} and {candidate.name}"
                )
            found[index] = candidate

        indices = sorted(found)
        if indices != list(range(1, len(indices) + 1)):
            missing = sorted(set(range(1, max(indices) + 1)) - set(indices))
            raise LedgerIntegrityError(
                f"ledger segment chain is not contiguous: missing segment(s) {missing}, "
                f"found {indices}"
            )
        return [self.path] + [found[i] for i in indices]

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

    def _verify_segment(
        self,
        index: int,
        path: Path,
        records: list[dict[str, Any]],
        predecessor: Path,
    ) -> None:
        """Check one rotated segment's header against the segment it sealed."""
        headers = [i for i, rec in enumerate(records) if rec.get("type") == "segment_open"]
        if not headers:
            raise LedgerIntegrityError(
                f"segment {path.name} is missing its segment_open header "
                f"(rotation may have been interrupted before publication)"
            )
        if headers != [0]:
            raise LedgerIntegrityError(
                f"segment {path.name} has a segment_open header at position(s) "
                f"{headers}; it must appear exactly once, first"
            )

        header = records[0]
        if header.get("segment") != index:
            raise LedgerIntegrityError(
                f"segment {path.name} header claims segment {header.get('segment')}, "
                f"but its filename says {index}"
            )
        if header.get("prev_segment") != predecessor.name:
            raise LedgerIntegrityError(
                f"segment {path.name} header names predecessor "
                f"{header.get('prev_segment')!r}, but the chain says {predecessor.name!r}"
            )

        raw = predecessor.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if header.get("prev_sha256") != actual:
            raise LedgerIntegrityError(
                f"sealed segment {predecessor.name} was modified after it was sealed: "
                f"expected sha256 {header.get('prev_sha256')}, found {actual}"
            )
        if header.get("prev_bytes") != len(raw):
            raise LedgerIntegrityError(
                f"sealed segment {predecessor.name} changed length after it was sealed: "
                f"expected {header.get('prev_bytes')} bytes, found {len(raw)}"
            )

    @staticmethod
    def _positive_int(value: Any) -> bool:
        """True for a genuine positive integer (JSON `true` is not one)."""
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    @classmethod
    def _verify_seqs(cls, records: list[dict[str, Any]]) -> None:
        """`seq` must be a positive integer, strictly increasing across the chain.

        Gaps are fine — a record lost to a crash consumed no durable seq, and a
        sealed fragment may have. Repeats and reversals are not: tombstones
        identify their target by `seq`, so a duplicate makes rollback ambiguous.

        The type check comes before the comparison on purpose. Disk data is
        untrusted input, and `"3" <= 2` raises `TypeError` — an incidental
        crash that says nothing about the ledger. Malformed data must fail as
        `LedgerIntegrityError` like every other chain violation.
        """
        previous: int | None = None
        for rec in records:
            if "seq" not in rec:
                raise LedgerIntegrityError(f"ledger record is missing 'seq': {rec!r}")
            seq = rec["seq"]
            if not cls._positive_int(seq):
                raise LedgerIntegrityError(
                    f"ledger record has a non-positive-integer 'seq' {seq!r}: {rec!r}"
                )
            if previous is not None and seq <= previous:
                raise LedgerIntegrityError(
                    f"ledger seq is not strictly increasing: {seq} follows {previous}"
                )
            previous = seq

    @classmethod
    def _verify_controls(cls, records: list[dict[str, Any]]) -> None:
        """Check every tombstone against the record it claims to supersede.

        A `redo` is the only way a durable record leaves active work, so an
        unchecked one is a delete in disguise. Read verification therefore
        proves each tombstone refers to a real, earlier, still-active unit of
        work belonging to the same prompt. Anything else — a forged target, a
        target that is itself a control record, a second tombstone for an
        already-superseded record — means the chain no longer describes what
        was actually mined, and the ledger fails closed.
        """
        work_by_seq: dict[int, dict[str, Any]] = {}
        superseded: dict[int, dict[str, Any]] = {}

        for rec in records:
            rec_type = rec.get("type")
            if rec_type not in CONTROL_TYPES:
                work_by_seq[rec["seq"]] = rec
                continue
            if rec_type != "redo":
                continue

            target_seq = rec.get("supersedes_seq")
            if not cls._positive_int(target_seq):
                raise LedgerIntegrityError(
                    f"redo record has a non-positive-integer 'supersedes_seq' "
                    f"{target_seq!r}: {rec!r}"
                )
            if target_seq >= rec["seq"]:
                raise LedgerIntegrityError(
                    f"redo at seq {rec['seq']} supersedes seq {target_seq}, which is "
                    f"not an earlier record"
                )
            target = work_by_seq.get(target_seq)
            if target is None:
                raise LedgerIntegrityError(
                    f"redo at seq {rec['seq']} supersedes seq {target_seq}, which is "
                    f"not an earlier work record in this chain"
                )
            if "prompt_id" not in rec:
                raise LedgerIntegrityError(
                    f"redo record is missing required field 'prompt_id': {rec!r}"
                )
            if rec["prompt_id"] != target["prompt_id"]:
                raise LedgerIntegrityError(
                    f"redo at seq {rec['seq']} names prompt_id {rec['prompt_id']!r} but "
                    f"seq {target_seq} recorded {target['prompt_id']!r}"
                )
            if target_seq in superseded:
                raise LedgerIntegrityError(
                    f"seq {target_seq} is superseded twice: by the redo at seq "
                    f"{superseded[target_seq]['seq']} and again at seq {rec['seq']}"
                )
            superseded[target_seq] = rec

    # ---------------------------------------------------------------- read --

    def _read_records(self) -> tuple[list[dict[str, Any]], bool]:
        """Parse and verify every complete line across the whole segment chain.

        Returns the parsed records in chain order and whether any segment ends
        in a truncated fragment. Fragments reflect work that was never durably
        recorded and are simply redone by the caller.

        Raises `LedgerIntegrityError` if the chain does not verify — the ledger
        fails closed rather than reporting a state it cannot vouch for.
        """
        segments = self.segment_paths()
        parsed = [self._parse_segment(segment) for segment in segments]

        # The base segment opens the chain, so it seals nothing and must carry
        # no header. A `segment_open` here would claim a predecessor that
        # cannot exist, and the per-segment check below never looks at it.
        base_headers = [
            i for i, rec in enumerate(parsed[0][0]) if rec.get("type") == "segment_open"
        ]
        if base_headers:
            raise LedgerIntegrityError(
                f"base segment {segments[0].name} has a segment_open header at "
                f"position(s) {base_headers}; only rotated segments have headers"
            )

        for index in range(1, len(segments)):
            self._verify_segment(
                index, segments[index], parsed[index][0], segments[index - 1]
            )

        records: list[dict[str, Any]] = []
        truncated_tail = False
        for segment_records, segment_truncated in parsed:
            records.extend(segment_records)
            truncated_tail = truncated_tail or segment_truncated

        self._verify_seqs(records)
        self._verify_controls(records)
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

    @staticmethod
    def _fsync_dir(directory: Path) -> None:
        """Force the directory entry itself to durable storage."""
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

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

    def _publish_segment(self, path: Path, header: dict[str, Any]) -> None:
        """Create a new segment containing exactly `header`, atomically.

        The segment is built complete in a same-directory temp file and fsynced
        before it is linked into place, so the published path never exists in a
        partial state. `os.link` fails rather than clobbering if something is
        already there, which turns a double-rotation into an error instead of
        silent data loss.
        """
        directory = path.parent
        line = json.dumps(header, sort_keys=True) + "\n"

        fd, tmp_name = tempfile.mkstemp(dir=str(directory), prefix=f".{path.name}.", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(tmp, path)
            except FileExistsError as exc:
                raise LedgerIntegrityError(
                    f"cannot publish ledger segment {path.name}: it already exists"
                ) from exc
            self._fsync_dir(directory)
        finally:
            tmp.unlink(missing_ok=True)

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
        new_index = len(segments)

        header = {
            "type": "segment_open",
            "segment": new_index,
            "prev_segment": sealed.name,
            "prev_sha256": hashlib.sha256(raw).hexdigest(),
            "prev_bytes": len(raw),
            "reason": "truncated tail in previous segment (crash during write)",
            "seq": self._next_seq(),
        }
        self._publish_segment(self._segment_path(new_index), header)
        return self._segment_path(new_index)

    def _active_segment(self) -> Path:
        """The segment to append to, rotating first if the current one is torn."""
        segments = self.segment_paths()
        active = segments[-1]
        _, truncated_tail = self._parse_segment(active)
        if truncated_tail:
            return self._seal_and_rotate()
        return active

    def _commit(self, record: dict[str, Any]) -> dict[str, Any]:
        """The sole write path for every record type.

        Order matters and is the point of this method: rotate first, so any
        segment header is already durable; *then* allocate `seq` against the
        post-rotation chain; then write. Allocating before rotating hands the
        header and the record the same number.
        """
        target = self._active_segment()
        out = dict(record)
        out["seq"] = self._next_seq()
        self._write_line(target, out)
        return out

    def append(self, record: dict[str, Any]) -> None:
        """Append one record of completed work, assigning the next monotonic `seq`.

        Raises `ValueError` if `prompt_id` is missing, if `prompt_id` is
        already active (appended and not since rolled back via `redo_last`) —
        this is the dedup guard that keeps a prompt from being double-counted
        across resumed runs — or if the caller tries to write a control record.

        Control records are the ledger's own bookkeeping, not units of work.
        Letting a caller write one through the public API would let ordinary
        mining code retire a record from active work (`redo`) or fabricate a
        seal over history it did not write (`segment_open`) — an edit to the
        evidence, dressed as an append. Only `redo_last` and `_seal_and_rotate`
        may create them, and both go through `_commit` directly.
        """
        record_type = record.get("type")
        if record_type in CONTROL_TYPES:
            raise ValueError(
                f"type {record_type!r} is reserved for ledger control records and "
                f"cannot be appended by a caller"
            )
        if "prompt_id" not in record:
            raise ValueError("record is missing required field 'prompt_id'")

        prompt_id = record["prompt_id"]
        if prompt_id in self.processed_ids():
            raise ValueError(f"prompt_id {prompt_id!r} was already appended and not rolled back")

        self._commit(record)

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
            # `seq` is deliberately not set here: _commit allocates it after
            # any rotation, so a tombstone can never share a number with the
            # segment header that rotation wrote.
            self._commit(
                {
                    "type": "redo",
                    "supersedes_seq": rec["seq"],
                    "prompt_id": rec["prompt_id"],
                }
            )

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
