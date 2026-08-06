"""Assign each pool prompt to a mining stratum by the number of tools it presents.

The stratum is the unit the yield gate standardizes over (prereg §2.2, §2.5), so
"how many tools does this prompt offer" has to be a committed, deterministic
function rather than a number someone once counted. It was not: §1's provisional
`8,117 multi-tool` came from a count that could only read one of the two prompt
formats in the pool, and silently classified the other 1,161 rows as not
multi-tool. This module exists so that figure is reproducible, testable, and
wrong in public if it is wrong.

Two formats are recognised because the pool carries two:

    xlam-style     "Tools:\\n[ {...}, {...} ]"
    hermes-style   "<tools>\\n[ {...}, {...} ]\\n</tools>"

A prompt is `ineligible` if its tool list parses under neither format, or if it
parses to zero tools — readable, but with nothing for the model to call. Either
way it is *not* mined and enters neither yield term; it is recorded here, in the
pre-mining eligibility artifact, rather than as a mining-ledger record, since an
active ledger record means one unit of completed work and would land in A2.1's
denominator.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

MULTI = "multi"
SINGLE = "single"
INELIGIBLE = "ineligible"

# Why a prompt is ineligible. Both are excluded from the yield terms, but they
# are different facts and a receipt that merged them would hide one: a prompt
# whose tool list we cannot read is a parser gap, while a prompt offering no
# tools is readable and simply cannot produce a tool-call pair.
NO_TOOL_LIST = "no_tool_list"
ZERO_TOOLS = "zero_tools"

# Non-greedy, and requires a bracketed array: the hermes system prompt names the
# empty literal `<tools></tools>` in its own instructions, and a laxer pattern
# matches that first and parses nothing.
_HERMES = re.compile(r"<tools>\s*(\[.*?\])\s*</tools>", re.S)
_XLAM_MARKER = "Tools:"


def tool_count(system_prompt: str) -> int | None:
    """How many tools does this prompt present? `None` if neither format parses.

    Where more than one well-formed list is present, the one with the **most
    tools** wins: a prompt that embeds a small example block alongside its real
    tool list must not be scored on the example.
    """
    counts: list[int] = []

    marker = system_prompt.find(_XLAM_MARKER)
    if marker != -1:
        try:
            parsed = json.loads(system_prompt[marker + len(_XLAM_MARKER):].strip())
            if isinstance(parsed, list):
                counts.append(len(parsed))
        except (json.JSONDecodeError, ValueError):
            pass

    for match in _HERMES.finditer(system_prompt):
        try:
            parsed = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, list):
            counts.append(len(parsed))

    return max(counts) if counts else None


def stratum_of(system_prompt: str) -> tuple[str, str | None]:
    """Return `(stratum, ineligibility_reason)`.

    A prompt presenting zero tools is parseable and still ineligible: there is
    no tool for the model to call, so it can never contribute a pair. Letting a
    successfully parsed empty list fall through to `single` would put it in the
    yield denominator and in the weight for a stratum it is not in.
    """
    count = tool_count(system_prompt)
    if count is None:
        return INELIGIBLE, NO_TOOL_LIST
    if count < 1:
        return INELIGIBLE, ZERO_TOOLS
    return (MULTI if count >= 2 else SINGLE), None


def _system_prompt(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "system":
            return message.get("content", "")
    return ""


def composition(pool_path: Path) -> dict[str, Any]:
    """Stratum counts for a pool file, with the input's sha256 alongside them.

    The digest travels with the counts because the counts are only meaningful
    against a named set of bytes; a composition figure quoted without one is the
    provisional kind this module was written to replace.
    """
    payload = pool_path.read_bytes()
    counts = {MULTI: 0, SINGLE: 0, INELIGIBLE: 0}
    reasons = {NO_TOOL_LIST: 0, ZERO_TOOLS: 0}
    by_source: dict[str, dict[str, int]] = {}

    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stratum, reason = stratum_of(_system_prompt(row))
        counts[stratum] += 1
        if reason is not None:
            reasons[reason] += 1
        source = str(row.get("source", "unknown"))
        by_source.setdefault(source, {MULTI: 0, SINGLE: 0, INELIGIBLE: 0})[stratum] += 1

    classifiable = counts[MULTI] + counts[SINGLE]
    return {
        "pool_path": pool_path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": sum(counts.values()),
        "counts": counts,
        "ineligible_reasons": reasons,
        "classifiable": classifiable,
        "multi_share": (counts[MULTI] / classifiable) if classifiable else None,
        "by_source": by_source,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pool", type=Path)
    parser.add_argument("--json", type=Path, help="write the receipt as JSON")
    args = parser.parse_args()

    receipt = composition(args.pool)
    counts = receipt["counts"]
    print(f"{receipt['pool_path']}  sha256={receipt['sha256']}")
    print(f"  rows          {receipt['rows']}")
    print(f"  multi         {counts[MULTI]}")
    print(f"  single        {counts[SINGLE]}")
    print(f"  ineligible    {counts[INELIGIBLE]}  {receipt['ineligible_reasons']}")
    if receipt["multi_share"] is not None:
        print(f"  multi share   {receipt['multi_share'] * 100:.1f}% of classifiable")
    for source, breakdown in sorted(receipt["by_source"].items()):
        print(f"    {source:<10} {breakdown}")

    if args.json:
        args.json.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
