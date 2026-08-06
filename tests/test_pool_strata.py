"""Tests for the mining stratum assignment.

The stratum decides which yield term a prompt lands in, and the composition
decides the gate's target weights, so the failure that matters here is the quiet
one: a parser that reads only the format it was written against and reports the
rest as "not multi-tool" rather than as unread. That is exactly how §1's
provisional 8,117 came to be wrong, so both formats are tested, and so is the
prompt that satisfies neither.
"""

from __future__ import annotations

import json
from pathlib import Path

from mining.pool_strata import (
    INELIGIBLE,
    MULTI,
    NO_TOOL_LIST,
    SINGLE,
    ZERO_TOOLS,
    composition,
    stratum_of,
    tool_count,
)

XLAM_TWO = 'You have tools.\n\nTools:\n[{"name": "a"}, {"name": "b"}]'
XLAM_ONE = 'You have tools.\n\nTools:\n[{"name": "only"}]'
HERMES_THREE = (
    "You are a function calling AI model. Signatures are within <tools> </tools> XML tags.\n"
    '<tools>\n[{"name": "a"}, {"name": "b"}, {"name": "c"}]\n</tools>'
)
HERMES_ONE = 'signatures within <tools></tools>\n<tools>\n[{"name": "solo"}]\n</tools>'


def test_xlam_format_is_counted() -> None:
    assert tool_count(XLAM_TWO) == 2
    assert stratum_of(XLAM_TWO) == (MULTI, None)
    assert stratum_of(XLAM_ONE) == (SINGLE, None)


def test_hermes_format_is_counted() -> None:
    """The regression that produced the wrong composition figure: these rows
    parsed under no rule at all and were silently counted as single-tool."""
    assert tool_count(HERMES_THREE) == 3
    assert stratum_of(HERMES_THREE) == (MULTI, None)
    assert stratum_of(HERMES_ONE) == (SINGLE, None)


def test_the_empty_tools_literal_in_the_instructions_is_not_the_tool_list() -> None:
    """Hermes prompts name `<tools></tools>` while explaining the format. A
    laxer pattern matches that first, parses nothing, and reports the prompt as
    unreadable."""
    assert tool_count(HERMES_ONE) == 1


def test_a_prompt_with_no_recognisable_tool_list_is_ineligible() -> None:
    assert tool_count("You are a helpful assistant. No tools here.") is None
    assert stratum_of("You are a helpful assistant.") == (INELIGIBLE, NO_TOOL_LIST)


def test_malformed_json_is_unparseable_not_zero() -> None:
    """A truncated list must not read as a tool list of length zero, which would
    land in `single` and quietly move a weight."""
    assert tool_count('Tools:\n[{"name": "a"},') is None


def test_the_largest_well_formed_list_wins() -> None:
    """A prompt carrying an example block alongside its real one is scored on
    the real one."""
    both = 'Tools:\n[{"name": "a"}]\n<tools>[{"name": "x"}, {"name": "y"}]</tools>'
    assert tool_count(both) == 2


def test_composition_reports_counts_with_the_input_digest(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    rows = [
        {"source": "xlam", "messages": [{"role": "system", "content": XLAM_TWO}]},
        {"source": "xlam", "messages": [{"role": "system", "content": XLAM_ONE}]},
        {"source": "hermes", "messages": [{"role": "system", "content": HERMES_THREE}]},
        {"source": "hermes", "messages": [{"role": "system", "content": "no tools"}]},
    ]
    pool.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    receipt = composition(pool)

    assert receipt["counts"] == {MULTI: 2, SINGLE: 1, INELIGIBLE: 1}
    assert receipt["classifiable"] == 3
    assert receipt["multi_share"] == 2 / 3
    assert receipt["by_source"]["hermes"][MULTI] == 1
    assert len(receipt["sha256"]) == 64


def test_the_digest_changes_with_the_pool(tmp_path: Path) -> None:
    """The composition is only meaningful against named bytes, so the receipt
    must not survive an edit to the file it describes."""
    pool = tmp_path / "pool.jsonl"
    row = {"source": "xlam", "messages": [{"role": "system", "content": XLAM_TWO}]}
    pool.write_text(json.dumps(row) + "\n")
    first = composition(pool)["sha256"]

    pool.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    assert composition(pool)["sha256"] != first


def test_an_empty_tool_list_is_ineligible_not_single() -> None:
    """A parsed empty list is readable and still has nothing to call. Letting it
    fall through to `single` would put a prompt that can never yield a pair into
    the yield denominator, and into the weight for a stratum it is not in."""
    for prompt in ("Tools:\n[]", "<tools>[]</tools>"):
        assert tool_count(prompt) == 0
        assert stratum_of(prompt) == (INELIGIBLE, ZERO_TOOLS)


def test_the_receipt_separates_why_a_prompt_was_ineligible(tmp_path: Path) -> None:
    """Unreadable and zero-tool are both excluded but are different facts: one
    is a parser gap worth fixing, the other is the data being what it is."""
    pool = tmp_path / "pool.jsonl"
    rows = [
        {"source": "x", "messages": [{"role": "system", "content": "Tools:\n[]"}]},
        {"source": "x", "messages": [{"role": "system", "content": "nothing here"}]},
        {"source": "x", "messages": [{"role": "system", "content": XLAM_TWO}]},
    ]
    pool.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    receipt = composition(pool)

    assert receipt["counts"][INELIGIBLE] == 2
    assert receipt["ineligible_reasons"] == {NO_TOOL_LIST: 1, ZERO_TOOLS: 1}
    assert receipt["classifiable"] == 1
