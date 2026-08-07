"""Tests for the cumulative stage spend ledger.

The failure this file exists for: a per-invocation timer means a stage that
crashes four times bills four full allowances while every individual run looks
compliant. The allowance is a property of the stage.
"""

from __future__ import annotations

import pytest

from mining.spend_ledger import (
    SpendLedgerError,
    close_session,
    consumed_seconds,
    open_session,
)


def test_a_resume_gets_what_is_left_not_a_fresh_window(tmp_path) -> None:
    session, first = open_session(tmp_path, 1000, 0.53, 1.00, now=0.0)
    assert first.consumed_seconds == 0
    close_session(tmp_path, session, now=400.0, exit_reason="stopped", rate_usd=0.53)

    _second_session, second = open_session(tmp_path, 1000, 0.53, 1.00, now=500.0)

    assert second.consumed_seconds == pytest.approx(400.0)
    assert second.remaining_seconds == pytest.approx(600.0)
    assert second.sessions == 1


def test_an_exhausted_stage_cannot_be_restarted(tmp_path) -> None:
    session, _ = open_session(tmp_path, 100, 0.53, 1.00, now=0.0)
    close_session(tmp_path, session, now=150.0, exit_reason="stopped", rate_usd=0.53)

    with pytest.raises(SpendLedgerError, match="does not grant a new one"):
        open_session(tmp_path, 100, 0.53, 1.00, now=200.0)


def test_a_hard_kill_is_not_cheaper_than_a_clean_stop(tmp_path) -> None:
    """A session with no end record is charged to the next start, so killing the
    process cannot launder the time the pod was held."""
    open_session(tmp_path, 10_000, 0.53, 1.00, now=0.0)  # never closed
    open_session(tmp_path, 10_000, 0.53, 1.00, now=300.0)

    consumed, sessions = consumed_seconds(tmp_path, now=350.0)

    assert sessions == 2
    assert consumed == pytest.approx(350.0), "300s orphaned + 50s live"


def test_the_live_session_counts_while_it_runs(tmp_path) -> None:
    open_session(tmp_path, 1000, 0.53, 1.00, now=100.0)

    consumed, _ = consumed_seconds(tmp_path, now=175.0)

    assert consumed == pytest.approx(75.0)


def test_the_close_record_carries_the_stage_cost(tmp_path) -> None:
    session, _ = open_session(tmp_path, 10_000, 0.53, 1.00, now=0.0)

    record = close_session(
        tmp_path, session, now=3600.0, exit_reason="completed", rate_usd=0.53
    )

    assert record["stage_consumed_seconds"] == pytest.approx(3600.0)
    assert record["stage_estimated_cost_usd"] == pytest.approx(0.53)
    assert record["exit_reason"] == "completed"


def test_an_empty_directory_has_consumed_nothing(tmp_path) -> None:
    assert consumed_seconds(tmp_path, now=10.0) == (0.0, 0)
