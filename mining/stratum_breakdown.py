"""Derive the per-stratum bucket breakdown from a mining ledger.

Committed because a hand-entered table is a claim; a script plus the ledger's
digest is evidence. The artifact this writes carries the digest of the ledger it
was derived from, so `pytest` can re-derive it and fail if either drifts.

Reads only **active** records — the same `seq`-based selection `summarize()`
uses — so a rolled-back prompt leaves this breakdown exactly as it leaves yield.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mining.ledger import CONTROL_TYPES, Ledger

BUCKETS = ("pair", "all_correct", "zero_correct")
STRATA = ("multi", "single")


def bucket_of(verdicts: list[dict[str, Any]]) -> str:
    accepted = sum(1 for v in verdicts if v["accepted"])
    if accepted == 0:
        return "zero_correct"
    if accepted == len(verdicts):
        return "all_correct"
    return "pair"


def breakdown(ledger_path: Path) -> dict[str, Any]:
    records = Ledger(ledger_path).records()
    superseded = {r["supersedes_seq"] for r in records if r.get("type") == "redo"}
    active = [
        r for r in records
        if r.get("type") not in CONTROL_TYPES and r["seq"] not in superseded
    ]

    counts = {b: dict.fromkeys(STRATA, 0) for b in BUCKETS}
    for record in active:
        counts[bucket_of(record["verdicts"])][record["stratum"]] += 1

    totals = {s: sum(counts[b][s] for b in BUCKETS) for s in STRATA}
    rates = {
        s: {b: (round(counts[b][s] / totals[s], 4) if totals[s] else None) for b in BUCKETS}
        for s in STRATA
    }
    return {
        "source_ledger": str(ledger_path),
        "source_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "active_records": len(active),
        "by_bucket": {
            b: {**counts[b], "total": sum(counts[b].values())} for b in BUCKETS
        },
        "stratum_totals": totals,
        "within_stratum_rates": rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = breakdown(args.ledger)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
