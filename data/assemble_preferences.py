"""Build DPO preference pairs from the deduped SFT set.

For each xLAM-source example in `sft_dedup.jsonl`, generate one preference
pair: `chosen` = the original correct assistant tool call, `rejected` = an
adversarially-perturbed variant. See ADR-004 for the full rationale and
perturbation taxonomy.

Output format
-------------
Each line is a JSON object with the shape:

    {
      "prompt_messages": [ system, user ],
      "chosen": "<original assistant content>",
      "rejected": "<perturbed assistant content>",
      "perturbation_type": "wrong_tool_from_list" | "hallucinated_tool" | ...,
      "source": "xlam",
      "source_id": "xlam-<original-id>"
    }

Design notes
------------
- Preference data schema follows TRL's DPOTrainer expectations: `chosen`
  and `rejected` are string completions of the same `prompt_messages`.
- Hermes examples are skipped per ADR-004 (their tool calls are embedded
  in prose; reliable rule-based perturbation isn't feasible).
- Deterministic: fixed seed so perturbation choices are reproducible from
  a git commit.

Usage
-----
    python data/assemble_preferences.py

Input:  data/processed/sft_dedup.jsonl
Output: data/processed/preferences.jsonl
"""

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

INPUT_PATH = Path("data/processed/sft_dedup.jsonl")
OUTPUT_PATH = Path("data/processed/preferences.jsonl")
SEED = 42

# Names likely to look plausible but aren't real tools. Used for
# `hallucinated_tool` perturbation.
PLAUSIBLE_FAKE_TOOLS = [
    "get_data",
    "fetch_info",
    "run_query",
    "call_api",
    "get_details_v2",
    "list_items",
    "search_records",
    "invoke_service",
]

# Which perturbation types are enabled. All five per ADR-004.
PERTURBATIONS = [
    "wrong_tool_from_list",
    "hallucinated_tool",
    "missing_required_arg",
    "wrong_arg_value",
    "malformed_json",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def extract_tools_from_system(system_content: str) -> list[dict[str, Any]]:
    """Parse the tools JSON out of an xLAM-format system prompt.

    xLAM system prompts (built by assemble_sft.py::build_system_prompt) have
    a `Tools:\\n<json>` section. We locate the JSON and parse it.
    """
    m = re.search(r"Tools:\s*(\[.*\])", system_content, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return []


def get_tool_arg_names(tool: dict[str, Any]) -> list[str]:
    """Extract the names of all declared arguments for a tool schema."""
    # xLAM tool schemas are OpenAI-style: {"name": ..., "parameters": {...}}
    params = tool.get("parameters", {})
    props = params.get("properties", {})
    return list(props.keys()) if isinstance(props, dict) else []


def get_tool_required_args(tool: dict[str, Any]) -> list[str]:
    """Extract the required argument names for a tool schema."""
    params = tool.get("parameters", {})
    required = params.get("required", [])
    return list(required) if isinstance(required, list) else []


def parse_assistant_tool_call(content: str) -> tuple[dict[str, Any], list | None] | None:
    """Parse the assistant's tool call. Returns (call, rest_of_list) or None.

    Content may be a single call `{...}` or a list of calls `[...]`. In the
    list case, we return (first_call, remaining_calls) so the caller can
    perturb just the first and preserve the rest in the rejected variant.
    Returns None if unparsable.
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list):
        if not parsed or not isinstance(parsed[0], dict) or "name" not in parsed[0]:
            return None
        return (parsed[0], parsed[1:])
    if not isinstance(parsed, dict) or "name" not in parsed:
        return None
    return (parsed, None)


# --------------------------------------------------------------------------
# Perturbation functions — each returns (perturbed_content, ok) or (None, False)
# --------------------------------------------------------------------------

def _wrong_tool_from_list(call: dict, tools: list[dict], rng: random.Random) -> str | None:
    other_names = [t["name"] for t in tools if t.get("name") and t["name"] != call["name"]]
    if not other_names:
        return None
    perturbed = dict(call)
    perturbed["name"] = rng.choice(other_names)
    return json.dumps(perturbed)


def _hallucinated_tool(call: dict, tools: list[dict], rng: random.Random) -> str | None:
    existing = {t.get("name") for t in tools}
    candidates = [n for n in PLAUSIBLE_FAKE_TOOLS if n not in existing]
    if not candidates:
        return None
    perturbed = dict(call)
    perturbed["name"] = rng.choice(candidates)
    return json.dumps(perturbed)


def _missing_required_arg(call: dict, tools: list[dict], rng: random.Random) -> str | None:
    args = dict(call.get("arguments") or {})
    if not args:
        return None
    # Prefer dropping a schema-declared required arg if we can find one.
    # Fall back to dropping any provided arg — an arg the model chose to
    # emit is a plausible-missing-arg failure mode regardless of whether
    # the schema explicitly marks it required.
    tool = next((t for t in tools if t.get("name") == call["name"]), None)
    droppable: list[str] = []
    if tool:
        required = get_tool_required_args(tool)
        droppable = [a for a in required if a in args]
    if not droppable:
        droppable = list(args.keys())
    if not droppable:
        return None
    to_drop = rng.choice(droppable)
    del args[to_drop]
    perturbed = dict(call)
    perturbed["arguments"] = args
    return json.dumps(perturbed)


def _wrong_arg_value(call: dict, tools: list[dict], rng: random.Random) -> str | None:
    args = dict(call.get("arguments") or {})
    if not args:
        return None
    arg_name = rng.choice(list(args.keys()))
    value = args[arg_name]
    # Simple perturbations by value type.
    if isinstance(value, str) and value:
        # Reverse the string — obviously wrong but syntactically valid.
        args[arg_name] = value[::-1]
    elif isinstance(value, bool):
        args[arg_name] = not value
    elif isinstance(value, (int, float)):
        args[arg_name] = -abs(value) - 999 if value >= 0 else abs(value) + 999
    elif isinstance(value, list):
        args[arg_name] = []
    else:
        return None
    perturbed = dict(call)
    perturbed["arguments"] = args
    return json.dumps(perturbed)


def _malformed_json(call: dict, tools: list[dict], rng: random.Random) -> str | None:
    good = json.dumps(call)
    # Pick a corruption strategy.
    strategy = rng.choice(["trailing_comma", "unterminated_string", "missing_brace"])
    if strategy == "trailing_comma":
        # Insert a trailing comma before the closing brace.
        idx = good.rfind("}")
        return good[:idx] + "," + good[idx:] if idx > 0 else None
    if strategy == "unterminated_string":
        # Find the first `"` and drop its closing partner in the value.
        m = re.search(r'":\s*"([^"]+)"', good)
        if not m:
            return None
        # Remove the closing quote at that position.
        return good[: m.end() - 1] + good[m.end() :]
    if strategy == "missing_brace":
        idx = good.rfind("}")
        return good[:idx] if idx > 0 else None
    return None


PERTURBATION_FUNCS = {
    "wrong_tool_from_list": _wrong_tool_from_list,
    "hallucinated_tool": _hallucinated_tool,
    "missing_required_arg": _missing_required_arg,
    "wrong_arg_value": _wrong_arg_value,
    "malformed_json": _malformed_json,
}


def perturb(
    call: dict,
    tools: list[dict],
    rng: random.Random,
) -> tuple[str, str] | None:
    """Apply one perturbation, cycling through types if the first fails.

    Returns (perturbation_type_used, rejected_content) or None if no
    perturbation could be applied.
    """
    order = rng.sample(PERTURBATIONS, len(PERTURBATIONS))
    for p_type in order:
        result = PERTURBATION_FUNCS[p_type](call, tools, rng)
        if result is not None:
            return (p_type, result)
    return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(
            f"Input not found: {INPUT_PATH}\n"
            f"Run `python data/dedupe.py` first."
        )

    rng = random.Random(SEED)

    input_examples: list[dict] = []
    with open(INPUT_PATH) as f:
        for line in f:
            input_examples.append(json.loads(line))
    print(f"Loaded {len(input_examples)} examples")

    pairs: list[dict] = []
    skipped_non_xlam = 0
    skipped_unparsable_tools = 0
    skipped_unparsable_call = 0
    skipped_no_perturbation = 0
    perturbation_counts: Counter[str] = Counter()

    for ex in input_examples:
        # xLAM only per ADR-004
        if ex.get("source") != "xlam":
            skipped_non_xlam += 1
            continue

        messages = ex.get("messages", [])
        if len(messages) < 3:
            continue

        system_msg = messages[0]
        user_msg = messages[1]
        assistant_msg = messages[2]

        if system_msg.get("role") != "system" or user_msg.get("role") != "user":
            continue
        if assistant_msg.get("role") != "assistant":
            continue

        tools = extract_tools_from_system(system_msg.get("content", ""))
        if not tools:
            skipped_unparsable_tools += 1
            continue

        chosen_content = assistant_msg.get("content", "")
        parsed = parse_assistant_tool_call(chosen_content)
        if parsed is None:
            skipped_unparsable_call += 1
            continue
        call, rest = parsed

        result = perturb(call, tools, rng)
        if result is None:
            skipped_no_perturbation += 1
            continue
        p_type, perturbed_call_str = result

        # If original was a list (multi-call), reassemble the rejected as
        # [perturbed_first, ...original_rest] so the shape matches chosen.
        if rest is not None:
            try:
                perturbed_call = json.loads(perturbed_call_str)
                rejected_content = json.dumps([perturbed_call] + rest)
            except json.JSONDecodeError:
                # `malformed_json` perturbation intentionally breaks parse —
                # still valid to embed as a raw string in a list-shaped output.
                # Reassemble manually: replace the first element in the JSON.
                rejected_content = "[" + perturbed_call_str + "".join(
                    "," + json.dumps(r) for r in rest
                ) + "]"
        else:
            rejected_content = perturbed_call_str
        perturbation_counts[p_type] += 1

        pairs.append(
            {
                "prompt_messages": [
                    {"role": "system", "content": system_msg["content"]},
                    {"role": "user", "content": user_msg["content"]},
                ],
                "chosen": chosen_content,
                "rejected": rejected_content,
                "perturbation_type": p_type,
                "source": ex.get("source", "xlam"),
                "source_id": ex.get("source_id", ""),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"  Pairs generated:            {len(pairs)}")
    print(f"  Skipped (non-xLAM source):  {skipped_non_xlam}")
    print(f"  Skipped (tools unparsable): {skipped_unparsable_tools}")
    print(f"  Skipped (call unparsable):  {skipped_unparsable_call}")
    print(f"  Skipped (no perturbation):  {skipped_no_perturbation}")
    print("  Perturbation type distribution:")
    for p_type, count in perturbation_counts.most_common():
        pct = 100 * count / len(pairs) if pairs else 0
        print(f"    {p_type}: {count} ({pct:.1f}%)")
    print(
        "\nNext step: `python data/push_to_hf.py` to publish the "
        "preference dataset to HuggingFace (coming next commit)."
    )


if __name__ == "__main__":
    main()
