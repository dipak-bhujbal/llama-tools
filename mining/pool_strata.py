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

import ast
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

# The structural eligibility rule, versioned so a future revision recomputes
# membership against a named criterion rather than inheriting an id list.
ELIGIBILITY_CRITERION_ID = "pool-target-structural-eligibility/v1"
ELIGIBILITY_PREDICATE = (
    "A prompt-eligible row is structurally excluded when its assistant target announces "
    "tool-call or JSON syntax and cannot be parsed completely into the declared call form "
    "-- every tool-call marker pair accounted for, every block parsed, and every call "
    "carrying a non-empty top-level string `name`. Prose targets are `no_call` and are "
    "retained. Name mismatches are defects, not exclusions, and always fail closed."
)
ZERO_TOOLS = "zero_tools"

# Non-greedy, and requires a bracketed array: the hermes system prompt names the
# empty literal `<tools></tools>` in its own instructions, and a laxer pattern
# matches that first and parses nothing.
_HERMES = re.compile(r"<tools>\s*(\[.*?\])\s*</tools>", re.S)
_XLAM_MARKER = "Tools:"
# Some Hermes targets carry the tag boundary as the two characters backslash-n
# rather than a newline, so a plain `\s*` misses 61 rows in the current pool.
_TOOL_CALL = re.compile(r"<tool_call>(?:\s|\\[nrt])*(\{.*?\})(?:\s|\\[nrt])*</tool_call>", re.S)
_TOOL_CALL_MARKER = "<tool_call>"
_TOOL_CALL_OPEN = re.compile(r"<tool_call>")
_TOOL_CALL_CLOSE = re.compile(r"</tool_call>")


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


CALL = "call"
NO_CALL = "no_call"
UNREADABLE = "unreadable"


def classify_target(assistant_turn: str) -> tuple[str, set[str]]:
    """Return `(kind, names)` for what this example teaches the model to emit.

    Three outcomes, and the distinction is the point. `call` is a tool call and
    its names are checked against the prompt. `no_call` is a target that
    deliberately answers in prose -- a clarification request, a refusal to guess
    at missing arguments -- which is legitimate training signal and not a defect.
    `unreadable` is a target that announces itself as a tool call, or as JSON,
    and then does not parse; that is a hard failure, because a target we cannot
    read is one we cannot check, and treating it as benign is how an unchecked
    row becomes a passing row.
    """
    text = assistant_turn.strip()
    if not text:
        return UNREADABLE, set()

    if _TOOL_CALL_MARKER in text:
        blocks = _TOOL_CALL.findall(text)
        # Every opening marker must have produced a parsed block. One valid call
        # beside one malformed call is a malformed target, not a valid one --
        # 612 rows in the pinned pool carry more than one block, so accepting a
        # partial parse would silently check a fraction of them.
        if not (
            len(blocks)
            == len(_TOOL_CALL_OPEN.findall(text))
            == len(_TOOL_CALL_CLOSE.findall(text))
        ):
            return UNREADABLE, set()
        names: set[str] = set()
        for block in blocks:
            call = _parse_call_block(block)
            if call is None:
                return UNREADABLE, set()
            names.add(call)
        return (CALL, names) if names else (UNREADABLE, set())

    if text[0] in "[{":
        parsed_names = _json_call_names(text)
        return (CALL, parsed_names) if parsed_names else (UNREADABLE, set())

    # Prose. Not a call, and not a failure to read one.
    return NO_CALL, set()


def _parse_call_block(block: str) -> str | None:
    """The function name this `<tool_call>` block declares, or `None`.

    JSON first, then a safe Python-literal parse, because some targets embed a
    repr rather than JSON. Either way a **top-level string `name`** is required.

    There is deliberately no regex fallback. The previous one searched the block
    for `"name": "..."` and found it inside `arguments` on blocks that have no
    top-level name at all -- manufacturing a function name out of an ordinary
    argument key, and reporting 56 malformed targets as valid calls. A check that
    invents the value it is checking is worse than no check.
    """
    parsed: Any = None
    try:
        parsed = json.loads(block)
    except (json.JSONDecodeError, ValueError):
        try:
            parsed = ast.literal_eval(block)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return None
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name")
    return name if isinstance(name, str) and name else None


def _json_call_names(text: str) -> set[str]:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return set()
    calls = parsed if isinstance(parsed, list) else [parsed]
    names: set[str] = set()
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            return set()
        names.add(call["name"])
    return names


def target_names(assistant_turn: str) -> set[str] | None:
    """The tool names this example teaches the model to call. `None` if not a call."""
    kind, names = classify_target(assistant_turn)
    return names if kind == CALL else None


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


def target_defects(pool_path: Path) -> dict[str, Any]:
    """Preflight the pool's own targets: does every call name a presented tool?

    The answer-key rule of prereg A2.3, turned on our own training data.
    `simple_python_363` was a key expecting a name its item never offered; an SFT
    example whose assistant turn calls a tool absent from its own system prompt
    is that same defect on the training side, and it teaches the model to invent
    a tool name where no eval will attribute the habit back here.

    Fail-closed. A defect or an unreadable eligible target is a stop condition,
    not a row to drop: silently excluding them would repair the denominator by
    discarding the evidence that something is wrong.
    """
    payload = pool_path.read_bytes()
    defects: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    raw_rows = call_targets = no_call_targets = prompt_ineligible = 0

    for index, line in enumerate(payload.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        raw_rows += 1
        row = json.loads(line)
        system = _system_prompt(row)
        assistant = next(
            (m.get("content", "") for m in row.get("messages", []) if m.get("role") == "assistant"),
            "",
        )
        stratum, _ = stratum_of(system)
        if stratum == INELIGIBLE:
            # Not applicable rather than unchecked: the prompt itself is out of
            # the mining population, so its target is not a target we will use.
            prompt_ineligible += 1
            continue

        kind, called = classify_target(assistant)
        if kind == NO_CALL:
            no_call_targets += 1
            continue
        if kind == UNREADABLE:
            unreadable.append({"line": index + 1, "source_id": row.get("source_id")})
            continue

        call_targets += 1
        missing = sorted(called - presented_names(system))
        if missing:
            defects.append({
                "line": index + 1,
                "source_id": row.get("source_id"),
                "called_but_not_presented": missing,
            })

    eligible = call_targets + no_call_targets + len(unreadable)
    return {
        # Retained = what survives the exclusion rule. It exceeds `call_targets`
        # by exactly `no_call_targets`, because a prose target is a valid
        # training row that simply is not a call. Naming only one of the two
        # made a 10-row gap look like an unexplained category.
        "retained_rows": call_targets + no_call_targets,
        "excluded_rows": len(unreadable),
        "reconciliation": (
            f"eligible {eligible} = call {call_targets} + no_call {no_call_targets} "
            f"+ unreadable {len(unreadable)}; retained {call_targets + no_call_targets} "
            f"= eligible - unreadable; prompt_ineligible {prompt_ineligible} is outside "
            f"eligible entirely"
        ),
        "pool_path": pool_path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "raw_rows": raw_rows,
        "eligible_rows": eligible,
        "call_targets": call_targets,
        "no_call_targets": no_call_targets,
        "prompt_ineligible": prompt_ineligible,
        "criterion_id": ELIGIBILITY_CRITERION_ID,
        "criterion": ELIGIBILITY_PREDICATE,
        "structurally_excluded": len(unreadable),
        "structurally_excluded_rows": unreadable,
        "unreadable": len(unreadable),
        "unreadable_rows": unreadable,
        "defect_count": len(defects),
        "defects": defects,
        # Under the adopted exclusion rule the structurally-excluded rows are an
        # expected, declared output of the criterion -- not unhandled failures.
        # What must be clean is the retained population: its calls carry no
        # presented-name defect. A name mismatch still fails closed, because that
        # is a claim about a row we *can* read.
        "passed": not defects,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pool", type=Path)
    parser.add_argument("--json", type=Path, help="write the receipt as JSON")
    parser.add_argument("--targets", action="store_true",
                        help="run the target preflight instead of the composition count")
    args = parser.parse_args()

    if args.targets:
        report = target_defects(args.pool)
        print(f"{report['pool_path']}  sha256={report['sha256']}")
        for field in ("raw_rows", "eligible_rows", "call_targets", "no_call_targets",
                      "prompt_ineligible", "unreadable", "defect_count"):
            print(f"  {field:<18} {report[field]}")
        print(f"  {'PASSED':<18} {report['passed']}")
        if args.json:
            args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(f"wrote {args.json}")
        if not report["passed"]:
            raise SystemExit("target preflight FAILED: see defects/unreadable above")
        return

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
