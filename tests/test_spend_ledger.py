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
    consumed_usd,
    open_session,
)


def test_a_resume_gets_what_is_left_not_a_fresh_window(tmp_path) -> None:
    session, first = open_session(tmp_path, 0.92, 0.53, 1.00, now=0.0)
    assert first.consumed_usd == 0
    close_session(tmp_path, session, now=3600.0, exit_reason="stopped", rate_usd=0.53)

    _second, second = open_session(tmp_path, 0.92, 0.53, 1.00, now=3700.0)

    assert second.consumed_usd == pytest.approx(0.53)
    assert second.remaining_usd == pytest.approx(0.39)
    assert second.sessions == 1


def test_an_exhausted_stage_cannot_be_restarted(tmp_path) -> None:
    session, _ = open_session(tmp_path, 0.10, 0.53, 1.00, now=0.0)
    close_session(tmp_path, session, now=3600.0, exit_reason="stopped", rate_usd=0.53)

    with pytest.raises(SpendLedgerError, match="does not grant a new one"):
        open_session(tmp_path, 0.10, 0.53, 1.00, now=3700.0)


def test_a_cheaper_resume_cannot_widen_the_dollar_ceiling(tmp_path) -> None:
    """Codex's repro: 6000s at $0.53 then a resume at $0.33 granted 1200s more,
    taking the total to $0.9933 and eating the storage reserve."""
    session, _ = open_session(tmp_path, 0.92, 0.53, 1.00, now=0.0)
    close_session(tmp_path, session, now=6000.0, exit_reason="stopped", rate_usd=0.53)

    _s2, resumed = open_session(tmp_path, 0.92, 0.33, 1.00, now=6100.0)

    # 6000s at $0.53/hr = $0.8833 spent; $0.0367 remains, not another full window.
    assert resumed.consumed_usd == pytest.approx(0.8833, abs=1e-4)
    assert resumed.remaining_usd == pytest.approx(0.0367, abs=1e-4)
    assert resumed.remaining_seconds < 500


def test_a_resume_may_not_restate_the_ceiling(tmp_path) -> None:
    session, _ = open_session(tmp_path, 0.92, 0.53, 1.00, now=0.0)
    close_session(tmp_path, session, now=10.0, exit_reason="stopped", rate_usd=0.53)

    with pytest.raises(SpendLedgerError, match="may not restate the ceiling"):
        open_session(tmp_path, 5.00, 0.53, 6.00, now=20.0)


def test_a_truncated_tail_refuses_rather_than_resetting_the_total(tmp_path) -> None:
    """Codex's repro: the fragment was skipped, the next append concatenated
    onto it, the new session became invisible and consumed reset to (0, 0)."""
    from mining.spend_ledger import SPEND_LEDGER_NAME

    session, _ = open_session(tmp_path, 0.92, 0.53, 1.00, now=0.0)
    close_session(tmp_path, session, now=100.0, exit_reason="stopped", rate_usd=0.53)
    path = tmp_path / SPEND_LEDGER_NAME
    path.write_text(path.read_text() + '{"event": "session_st')

    with pytest.raises(SpendLedgerError, match="ends mid-record"):
        open_session(tmp_path, 0.92, 0.53, 1.00, now=200.0)


def test_a_hard_kill_is_not_cheaper_than_a_clean_stop(tmp_path) -> None:
    """A session with no end record is charged to the next start, so killing the
    process cannot launder the time the pod was held."""
    open_session(tmp_path, 0.92, 0.53, 1.00, now=0.0)  # never closed
    open_session(tmp_path, 0.92, 0.53, 1.00, now=300.0)

    _usd, sessions, seconds = consumed_usd(tmp_path, now=350.0)

    assert sessions == 2
    assert seconds == pytest.approx(350.0), "300s orphaned + 50s live"


def test_the_live_session_counts_while_it_runs(tmp_path) -> None:
    open_session(tmp_path, 0.92, 0.53, 1.00, now=100.0)

    _usd, _sessions, seconds = consumed_usd(tmp_path, now=175.0)

    assert seconds == pytest.approx(75.0)


def test_the_close_record_carries_the_stage_cost(tmp_path) -> None:
    session, _ = open_session(tmp_path, 0.92, 0.53, 1.00, now=0.0)

    record = close_session(
        tmp_path, session, now=3600.0, exit_reason="completed", rate_usd=0.53
    )

    assert record["stage_consumed_seconds"] == pytest.approx(3600.0)
    assert record["stage_cost_usd"] == pytest.approx(0.53)
    assert record["exit_reason"] == "completed"


def test_an_empty_directory_has_consumed_nothing(tmp_path) -> None:
    assert consumed_usd(tmp_path, now=10.0) == (0.0, 0, 0.0)
