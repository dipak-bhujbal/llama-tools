"""Tests for Tango's score confidence interval (prereg A1.4, §4.2).

Frozen A1.4 requires this interval for every reported contrast, and candidate
§4.2 recorded it as *not implemented* rather than assuming it existed. This file
is the evidence that closing that gap produced the right interval and not merely
a plausible one.

Two independent kinds of check, because either alone leaves a hole:

- **Conformance** against `PropCIs::scoreci.mp` — a third-party implementation
  that cites Tango (1998) — so a wrong *formula* is caught. Its outputs are
  pinned in `tests/fixtures/tango_propcis_oracle.json` with the package version,
  archive digest, and exact calls; these tests run offline and neither R nor
  PropCIs is a runtime or CI dependency.
- **Structural** checks — the endpoints solve the score equation, the interval
  brackets its own point estimate, swapping `b`/`c` mirrors it, and the
  documented degeneracies hold — so a right formula solved badly is caught.

The oracle is software conformance data. It is **not** a primary-paper
reproduction (Tango 1998 is paywalled and no accessible worked table was found)
and **not** study evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

import pytest

from eval.paired_analysis import compare, tango_interval, tango_score

REPO_ROOT = Path(__file__).resolve().parents[1]
ORACLE = json.loads((REPO_ROOT / "tests" / "fixtures" / "tango_propcis_oracle.json").read_text())
VECTORS = [(v["b"], v["c"], v["n"], v["lower"], v["upper"]) for v in ORACLE["vectors"]]

# The oracle's bisection stops at dp < 1e-7 or |z - score| < 1e-6, so this is the
# accuracy it can support. Asserting tighter would be claiming agreement the
# reference cannot actually witness.
ORACLE_TOLERANCE = 1e-6

Z_95 = NormalDist().inv_cdf(0.975)


@pytest.mark.parametrize(("b", "c", "n", "lower", "upper"), VECTORS)
def test_matches_the_propcis_oracle(b: int, c: int, n: int, lower: float, upper: float) -> None:
    mine_low, mine_high = tango_interval(b, c, n, conf_level=0.95)

    assert mine_low == pytest.approx(lower, abs=ORACLE_TOLERANCE)
    assert mine_high == pytest.approx(upper, abs=ORACLE_TOLERANCE)


@pytest.mark.parametrize(("b", "c", "n"), [(40, 20, 160), (13, 3, 400), (2, 2, 30), (1, 7, 50)])
def test_the_endpoints_solve_the_score_equation(b: int, c: int, n: int) -> None:
    """The interval is `{delta : |z(delta)| <= z_crit}`, so its endpoints are
    exactly where the score statistic hits ±z. This is what the oracle cannot
    check for us: it would agree with a formula that is wrong in the same way."""
    low, high = tango_interval(b, c, n)

    assert tango_score(b, c, n, low) == pytest.approx(Z_95, abs=1e-9)
    assert tango_score(b, c, n, high) == pytest.approx(-Z_95, abs=1e-9)


@pytest.mark.parametrize(("b", "c", "n"), [(40, 20, 160), (3, 13, 400), (0, 0, 100), (0, 5, 25)])
def test_the_interval_brackets_its_own_point_estimate(b: int, c: int, n: int) -> None:
    low, high = tango_interval(b, c, n)
    point = (c - b) / n

    assert low < point < high


@pytest.mark.parametrize(("b", "c", "n"), [(40, 20, 160), (13, 3, 400), (1, 7, 50), (0, 5, 25)])
def test_swapping_b_and_c_mirrors_the_interval(b: int, c: int, n: int) -> None:
    """`b` and `c` name which direction a disagreement went, so swapping them is
    a relabelling and must negate the interval exactly. A sign error that survives
    the oracle check would not survive this one."""
    low, high = tango_interval(b, c, n)
    swapped_low, swapped_high = tango_interval(c, b, n)

    assert swapped_low == pytest.approx(-high, abs=1e-12)
    assert swapped_high == pytest.approx(-low, abs=1e-12)


def test_zero_discordance_gives_a_symmetric_interval_around_zero() -> None:
    low, high = tango_interval(0, 0, 100)

    assert low == pytest.approx(-high, abs=1e-12)
    assert low < 0 < high


def test_the_direction_convention_is_candidate_minus_reference() -> None:
    """Stated in the docstring and asserted here, because a silent sign flip
    would reverse every reported result while every test of magnitude passed."""
    # Candidate wins the discordant items: interval must sit above zero.
    assert tango_interval(b=1, c=20, n=200)[0] > 0
    # Reference wins them: interval must sit below zero.
    assert tango_interval(b=20, c=1, n=200)[1] < 0


@pytest.mark.parametrize(("b", "c", "n"), [(50, 0, 50), (0, 50, 50)])
def test_the_documented_degeneracies_hold(b: int, c: int, n: int) -> None:
    """Every item discordant in one direction pins that endpoint at the boundary."""
    low, high = tango_interval(b, c, n)

    if b == n:
        assert low == -1.0
    if c == n:
        assert high == 1.0


def test_a_wider_confidence_level_gives_a_wider_interval() -> None:
    narrow = tango_interval(13, 3, 400, conf_level=0.90)
    wide = tango_interval(13, 3, 400, conf_level=0.99)

    assert wide[0] < narrow[0] < narrow[1] < wide[1]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"b": 1, "c": 1, "n": 0}, "n must be positive"),
        ({"b": -1, "c": 1, "n": 10}, "non-negative"),
        ({"b": 1, "c": -1, "n": 10}, "non-negative"),
        ({"b": 6, "c": 6, "n": 10}, "exceeds"),
        ({"b": 1, "c": 1, "n": 10, "conf_level": 0.0}, "conf_level"),
        ({"b": 1, "c": 1, "n": 10, "conf_level": 1.0}, "conf_level"),
    ],
)
def test_invalid_inputs_raise_rather_than_returning_a_number(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        tango_interval(**kwargs)


def test_compare_exposes_the_interval_for_every_contrast() -> None:
    by_candidate = {
        "sft": {"a": True, "b": True, "c": True, "d": False},
        "dpo": {"a": True, "b": False, "c": True, "d": True},
    }

    (row,) = compare(by_candidate, reference="sft")

    assert row["tango_ci_95"] == list(
        tango_interval(row["reference_only_correct"], row["candidate_only_correct"],
                       row["n_items"])
    )
    # An interval is not a test and is never multiplicity-adjusted (A1.4).
    assert "p_holm_adjusted" in row
    assert len(row["tango_ci_95"]) == 2


def test_the_oracle_fixture_states_what_it_is_and_is_not() -> None:
    """The fixture carries provenance because a bare table of numbers invites
    being cited as something it is not."""
    assert ORACLE["oracle"]["version"] == "0.3-0"
    assert ORACLE["oracle"]["archive_sha256"] == (
        "cd35775f4d36e642663e727450c53708f17b3c4340e1bc2c1752fd17118a9ffb"
    )
    assert "GPL" in ORACLE["oracle"]["license"]
    assert "NOT a reproduction of a primary-paper worked table" in ORACLE["what_this_is_not"]
    assert len(ORACLE["vectors"]) == 11
