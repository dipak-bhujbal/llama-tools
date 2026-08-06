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
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_CALL_NAME = re.compile(r'"name"\s*:\s*"([^"]+)"')


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


def presented_names(system_prompt: str) -> set[str]:
    """The tool names this prompt offers, under whichever format parses."""
    blocks: list[list[Any]] = []
    marker = system_prompt.find(_XLAM_MARKER)
    if marker != -1:
        try:
            parsed = json.loads(system_prompt[marker + len(_XLAM_MARKER):].strip())
            if isinstance(parsed, list):
                blocks.append(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
    for match in _HERMES.finditer(system_prompt):
        try:
            parsed = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, list):
            blocks.append(parsed)
    if not blocks:
        return set()
    chosen = max(blocks, key=len)
    names: set[str] = set()
    for tool in chosen:
        if isinstance(tool, dict):
            name = tool.get("name") or (tool.get("function") or {}).get("name")
            if isinstance(name, str):
                names.add(name)
    return names


def target_names(assistant_turn: str) -> set[str] | None:
    """The tool names this example teaches the model to call. `None` if unreadable."""
    blocks = _TOOL_CALL.findall(assistant_turn)
    if blocks:
        # Hermes targets embed a Python-repr dict often enough that json.loads
        # fails on the arguments; the tool *name* is still well-formed, and the
        # name is the whole of what this check needs. Falling back to it keeps
        # 1,161 rows inside the check rather than reporting them as unreadable
        # -- which would repeat, on the target side, exactly the blind spot the
        # presented-tool parser was fixed for.
        names: set[str] = set()
        for block in blocks:
            try:
                call = json.loads(block)
                if isinstance(call, dict) and isinstance(call.get("name"), str):
                    names.add(call["name"])
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
            found = _CALL_NAME.search(block)
            if found is None:
                return None
            names.add(found.group(1))
        return names or None

    try:
        parsed = json.loads(assistant_turn.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    calls = parsed if isinstance(parsed, list) else [parsed]
    names: set[str] = set()
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            return None
        names.add(call["name"])
    return names or None


def target_defects(pool_path: Path) -> dict[str, Any]:
    """Rows whose training target calls a tool the prompt never presented.

    The same rule the answer-key preflight applies to BFCL, turned on our own
    training pool: `simple_python_363` was a key expecting a name the item never
    offered, and an SFT example whose assistant turn calls a tool absent from its
    own system prompt is that defect on the training side. It teaches the model
    to invent a tool name, and no eval will attribute the habit back here.
    """
    payload = pool_path.read_bytes()
    defects: list[dict[str, Any]] = []
    checked = unreadable_target = 0

    for index, line in enumerate(payload.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        presented = presented_names(_system_prompt(row))
        assistant = next(
            (m.get("content", "") for m in row.get("messages", []) if m.get("role") == "assistant"),
            "",
        )
        called = target_names(assistant)
        if called is None:
            unreadable_target += 1
            continue
        checked += 1
        missing = sorted(called - presented)
        if missing:
            defects.append({
                "line": index + 1,
                "source_id": row.get("source_id"),
                "called_but_not_presented": missing,
            })

    return {
        "pool_path": pool_path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows_checked": checked,
        "unreadable_target": unreadable_target,
        "defect_count": len(defects),
        "defects": defects[:50],
    }


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
