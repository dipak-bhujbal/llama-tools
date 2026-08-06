"""Tests for `onpolicy_verifier_v1` (prereg §2.1, §2.11).

The verifier decides what counts as a policy failure, so every mined pair, the
yield feeding the §2.6 gate, and 3B's training set are downstream of it. Two
kinds of test therefore live here: the behaviour of each verdict, and proof that
the 1,600-pair gate can actually fail. A gate that cannot fail measures nothing,
and this repository has already been burned once by a "1,600/1,600" figure with
no run artifact behind it (`docs/HANDOFF.md` §A).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mining.verifier import (
    INVALID_JSON,
    MISSING_CALL,
    SPURIOUS_CALL,
    WRONG_ARGS,
    WRONG_TOOL,
    ParserDisagreementError,
    TargetUnreadableError,
    extract_calls,
    run_selftest,
    verify,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = (
    REPO_ROOT / "tests" / "fixtures" / "fixture_pairs_train.jsonl",
    REPO_ROOT / "tests" / "fixtures" / "fixture_pairs_eval.jsonl",
)

CALL = '{"name": "get_weather", "arguments": {"location": "Porto", "units": "celsius"}}'
PROSE = "It is overcast in Porto right now."


def test_an_exact_match_is_accepted() -> None:
    assert verify(CALL, CALL).accepted


def test_prose_against_a_call_target_is_a_missing_call() -> None:
    verdict = verify(PROSE, CALL)
    assert not verdict.accepted
    assert verdict.reason == MISSING_CALL


def test_a_call_against_a_prose_target_is_spurious() -> None:
    verdict = verify(CALL, PROSE)
    assert not verdict.accepted
    assert verdict.reason == SPURIOUS_CALL


def test_prose_against_a_prose_target_is_accepted() -> None:
    assert verify("Eleven players per side.", PROSE).accepted


def test_a_different_function_is_wrong_tool_not_wrong_args() -> None:
    other = '{"name": "get_forecast", "arguments": {"location": "Porto", "units": "celsius"}}'
    assert verify(other, CALL).reason == WRONG_TOOL


def test_same_function_different_arguments_is_wrong_args() -> None:
    other = '{"name": "get_weather", "arguments": {"location": "Lisbon", "units": "celsius"}}'
    assert verify(other, CALL).reason == WRONG_ARGS


def test_unparseable_generation_is_invalid_json_not_a_missing_call() -> None:
    """A generation that *announces* a call and fails to parse is a different
    failure from one that never tried; ADR-007 counts them separately."""
    assert verify('{"name" "get_weather"}', CALL).reason == INVALID_JSON


def test_argument_comparison_is_exact_with_no_type_coercion() -> None:
    target = '{"name": "sleep", "arguments": {"seconds": 5}}'
    stringy = '{"name": "sleep", "arguments": {"seconds": "5"}}'
    assert verify(stringy, target).reason == WRONG_ARGS


def test_multiple_calls_compare_order_insensitively() -> None:
    a = '{"name": "a", "arguments": {"x": 1}}'
    b = '{"name": "b", "arguments": {"y": 2}}'
    target = f"<tool_call>{a}</tool_call><tool_call>{b}</tool_call>"
    swapped = f"<tool_call>{b}</tool_call><tool_call>{a}</tool_call>"
    assert verify(swapped, target).accepted


def test_a_missing_argument_is_wrong_args_not_an_accept() -> None:
    partial = '{"name": "get_weather", "arguments": {"location": "Porto"}}'
    assert verify(partial, CALL).reason == WRONG_ARGS


# --- §2.11: refuse, never reclassify ---------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        "<tool_call>{not json}</tool_call>",
        "<tool_call>{}</tool_call>",
        '{"arguments": {"x": 1}}',
        "",
    ],
)
def test_an_unreadable_target_refuses_instead_of_becoming_a_no_call(target: str) -> None:
    """The quarantined miner's exact defect: an unparseable target silently
    became a `no_call` ground truth, so an unchecked row passed."""
    with pytest.raises(TargetUnreadableError):
        verify(CALL, target)


def test_the_two_parsers_must_agree_about_the_target(monkeypatch) -> None:
    """`classify_target` builds the committed eligibility receipt; this module
    reads arguments. If they disagree, one is defective and neither wins."""
    monkeypatch.setattr(
        "mining.verifier.classify_target", lambda _turn: ("call", {"something_else"})
    )
    with pytest.raises(ParserDisagreementError):
        verify(CALL, CALL)


def test_a_nested_name_never_manufactures_a_call() -> None:
    """The quarantined parser's regex found `"name"` inside `arguments`."""
    kind, calls = extract_calls('{"arguments": {"name": "get_weather"}}')
    assert (kind, calls) == ("unreadable", [])


# --- the gate itself --------------------------------------------------------


def test_the_fixture_gate_passes_on_the_committed_fixtures() -> None:
    report = run_selftest(FIXTURES)

    assert report.pairs == 1600
    assert report.pairs_passed == 1600
    assert report.misses == []
    assert report.false_positives == []
    assert report.reason_mismatches == []
    assert report.refusals == []
    assert report.passed


def test_the_gate_catches_a_rejected_sample_that_should_have_failed(tmp_path) -> None:
    """Proof the gate can fail. A pair whose `rejected` equals its `chosen` is a
    sample the verifier must accept — and accepting it is precisely the false
    positive the gate exists to catch."""
    row = {
        "prompt": [{"role": "user", "content": "weather?"}],
        "chosen": [{"role": "assistant", "content": CALL}],
        "rejected": [{"role": "assistant", "content": CALL}],
        "meta": {"pair_id": "control-1", "error_type": "wrong_param_value",
                 "synthetic": True},
    }
    path = tmp_path / "control.jsonl"
    path.write_text(json.dumps(row) + "\n")

    report = run_selftest([path])

    assert report.false_positives == ["control-1"]
    assert report.pairs_passed == 0
    assert not report.passed


def test_the_gate_catches_a_right_rejection_for_the_wrong_reason(tmp_path) -> None:
    """Rejecting is not enough: the reason must match the injected defect, or a
    verifier that is accidentally right would pass."""
    row = {
        "prompt": [{"role": "user", "content": "weather?"}],
        "chosen": [{"role": "assistant", "content": CALL}],
        "rejected": [{"role": "assistant", "content": PROSE}],
        # Labelled as an argument defect; the verifier will say missing_call.
        "meta": {"pair_id": "control-2", "error_type": "wrong_param_value",
                 "synthetic": True},
    }
    path = tmp_path / "control.jsonl"
    path.write_text(json.dumps(row) + "\n")

    report = run_selftest([path])

    assert report.false_positives == []
    assert [m["pair_id"] for m in report.reason_mismatches] == ["control-2"]
    assert not report.passed


def test_the_gate_reports_a_refusal_rather_than_scoring_around_it(tmp_path) -> None:
    row = {
        "prompt": [{"role": "user", "content": "weather?"}],
        "chosen": [{"role": "assistant", "content": "<tool_call>{not json}</tool_call>"}],
        "rejected": [{"role": "assistant", "content": PROSE}],
        "meta": {"pair_id": "control-3", "error_type": "missed_tool_call",
                 "synthetic": True},
    }
    path = tmp_path / "control.jsonl"
    path.write_text(json.dumps(row) + "\n")

    report = run_selftest([path])

    assert [r["pair_id"] for r in report.refusals] == ["control-3"]
    assert report.pairs_passed == 0
    assert not report.passed


def test_the_committed_selftest_receipt_reproduces() -> None:
    """The receipt is the run artifact behind the 1,600/1,600 figure. Without
    this, the number is exactly the unbacked claim §A of HANDOFF.md flags."""
    receipt = json.loads((REPO_ROOT / "mining" / "receipts" / "verifier_selftest.json").read_text())
    report = run_selftest(FIXTURES)

    assert receipt["verifier_version"] == "onpolicy_verifier_v1"
    assert receipt["pairs"] == report.pairs == 1600
    assert receipt["pairs_passed"] == report.pairs_passed == 1600
    assert receipt["passed"] is True
    assert all(fixture["synthetic"] is True for fixture in receipt["fixtures"])
