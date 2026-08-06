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
    CALL,
    INELIGIBLE,
    MULTI,
    NO_CALL,
    NO_TOOL_LIST,
    SINGLE,
    UNREADABLE,
    ZERO_TOOLS,
    classify_target,
    composition,
    presented_names,
    stratum_of,
    target_defects,
    target_names,
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


# ------------------------------------------------ the pool's own target preflight ----
#
# The answer-key preflight rule, turned on our own training data: an example
# whose assistant turn calls a tool its prompt never presented teaches the model
# to invent a tool name, and no eval attributes that habit back here.


def test_presented_names_reads_both_formats() -> None:
    assert presented_names(XLAM_TWO) == {"a", "b"}
    assert presented_names(HERMES_THREE) == {"a", "b", "c"}


def test_target_names_reads_a_json_call_list() -> None:
    assert target_names('[{"name": "a", "arguments": {}}]') == {"a"}
    assert target_names('{"name": "solo", "arguments": {}}') == {"solo"}


def test_target_names_reads_hermes_tool_call_blocks() -> None:
    turn = '<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n</tool_call>'
    assert target_names(turn) == {"a"}


def test_a_hermes_target_with_python_repr_arguments_still_yields_its_name() -> None:
    """These are not valid JSON -- single-quoted argument strings -- but the
    name is well-formed and the name is all this check needs. Reporting them
    unreadable would repeat the presented-side blind spot on the target side."""
    turn = "<tool_call>\n{\"name\": \"search\", \"arguments\": {'q': 'hi'}}\n</tool_call>"
    assert target_names(turn) == {"search"}


def test_target_defects_flags_a_call_to_an_unpresented_tool(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    rows = [
        {"source_id": "ok", "messages": [
            {"role": "system", "content": XLAM_TWO},
            {"role": "assistant", "content": '[{"name": "a", "arguments": {}}]'}]},
        {"source_id": "bad", "messages": [
            {"role": "system", "content": XLAM_TWO},
            {"role": "assistant", "content": '[{"name": "invented", "arguments": {}}]'}]},
    ]
    pool.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    report = target_defects(pool)

    assert report["call_targets"] == 2
    assert report["defect_count"] == 1
    assert report["passed"] is False
    assert report["defects"][0]["source_id"] == "bad"
    assert report["defects"][0]["called_but_not_presented"] == ["invented"]


def test_a_tool_call_tag_bounded_by_literal_backslash_n_is_read() -> None:
    """61 rows in the pool carry the tag boundary as the two characters
    backslash-n rather than a newline. A plain `\\s*` misses every one of them
    and reports a readable target as unreadable."""
    turn = '<tool_call>\\n{"name": "search", "arguments": {}}\\n</tool_call>'
    kind, names = classify_target(turn)
    assert (kind, names) == (CALL, {"search"})


def test_a_prose_target_is_no_call_not_unreadable() -> None:
    """Answering in prose -- asking for a missing argument rather than guessing
    -- is legitimate training signal, not a parse failure."""
    kind, names = classify_target("Certainly! Before proceeding I would need the data set.")
    assert (kind, names) == (NO_CALL, set())


def test_a_target_claiming_to_be_a_call_and_failing_to_parse_is_unreadable() -> None:
    """The hard failure: it announces itself as a tool call and then does not
    parse. A target we cannot read is one we cannot check."""
    assert classify_target("<tool_call>not json at all</tool_call>")[0] == UNREADABLE
    assert classify_target('[{"name": ')[0] == UNREADABLE


def test_the_preflight_fails_closed_on_an_unreadable_eligible_target(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    rows = [
        {"source_id": "ok", "messages": [
            {"role": "system", "content": XLAM_TWO},
            {"role": "assistant", "content": '[{"name": "a", "arguments": {}}]'}]},
        {"source_id": "broken", "messages": [
            {"role": "system", "content": XLAM_TWO},
            {"role": "assistant", "content": "<tool_call>garbage</tool_call>"}]},
    ]
    pool.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    report = target_defects(pool)

    assert report["unreadable"] == 1
    assert report["passed"] is False, "an unreadable eligible target must stop the freeze"


def test_a_prompt_ineligible_row_is_not_applicable_rather_than_a_pass(tmp_path: Path) -> None:
    """Its prompt is out of the mining population, so its target is not a target
    we will use -- counted separately, never as a checked pass."""
    pool = tmp_path / "pool.jsonl"
    rows = [
        {"source_id": "nopromptools", "messages": [
            {"role": "system", "content": "no tools here"},
            {"role": "assistant", "content": '[{"name": "x", "arguments": {}}]'}]},
    ]
    pool.write_text(json.dumps(rows[0]) + "\n")

    report = target_defects(pool)

    assert report["prompt_ineligible"] == 1
    assert report["call_targets"] == 0
    assert report["eligible_rows"] == 0
    assert report["passed"] is True


def test_one_valid_call_beside_one_malformed_call_is_unreadable() -> None:
    """612 rows in the pinned pool carry more than one tool-call block. A parser
    that accepts the first and ignores the rest checks a fraction of them and
    reports the whole row as clean."""
    turn = (
        '<tool_call>\n{"name": "ok", "arguments": {}}\n</tool_call>\n'
        "<tool_call>\nGARBAGE\n</tool_call>"
    )
    assert classify_target(turn)[0] == UNREADABLE


def test_an_unmatched_closing_tool_call_marker_is_unreadable() -> None:
    """Fail closed on a dangling close marker as well as a dangling open one."""
    turn = (
        '<tool_call>\n{"name": "ok", "arguments": {}}\n</tool_call>\n'
        "</tool_call>"
    )
    assert classify_target(turn)[0] == UNREADABLE


def test_a_block_without_a_top_level_name_is_unreadable() -> None:
    """The defect that invalidated the first receipt: these blocks carry only
    `arguments`, and a regex search for `"name"` finds an ordinary argument key
    nested inside it. Harvesting that manufactures a function name."""
    block = "{'arguments': {'queries': ['q'], 'name': 'ExpertQAExtractor'}}"
    turn = f"<tool_call>\n{block}\n</tool_call>"
    assert classify_target(turn)[0] == UNREADABLE


def test_a_python_repr_block_with_a_top_level_name_is_read() -> None:
    """The literal fallback is legitimate where the call is well-formed: a repr
    rather than JSON, but with a real top-level name."""
    turn = "<tool_call>\n{'name': 'search', 'arguments': {'q': 'hi'}}\n</tool_call>"
    assert classify_target(turn) == (CALL, {"search"})


def test_an_argument_literally_named_name_is_not_the_function_name() -> None:
    turn = '<tool_call>\n{"name": "real_fn", "arguments": {"name": "decoy"}}\n</tool_call>'
    assert classify_target(turn) == (CALL, {"real_fn"})


def test_the_receipt_preserves_every_failure_identity(tmp_path: Path) -> None:
    """A repair/exclusion decision needs the exact set, not a sampled prefix."""
    pool = tmp_path / "pool.jsonl"
    rows = [
        {
            "source_id": f"broken-{index}",
            "messages": [
                {"role": "system", "content": XLAM_TWO},
                {"role": "assistant", "content": "<tool_call>garbage</tool_call>"},
            ],
        }
        for index in range(56)
    ]
    pool.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    report = target_defects(pool)

    assert report["unreadable"] == 56
    assert len(report["unreadable_rows"]) == 56
    assert {row["source_id"] for row in report["unreadable_rows"]} == {
        f"broken-{index}" for index in range(56)
    }
