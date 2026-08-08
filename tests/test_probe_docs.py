"""The probe operator guide is load-bearing, so its numbers are tested.

`scripts/launch_probe.sh` derives its wall-clock `timeout` from `--usd-cap`. A
stale cap in the guide therefore does not produce an unenforced budget — it
produces a *mechanically enforced* one at the wrong number, which is worse,
because everything downstream looks correct. These tests exist because the guide
carried `$2.50` and a `5 h 00 m` deadline for hours after `$0.45` was approved.
"""

from __future__ import annotations

import re
from pathlib import Path

GUIDE = Path("docs/probe-bootstrap.md")
APPROVED_CAP = "0.45"


def _text() -> str:
    return GUIDE.read_text()


def test_no_superseded_dollar_cap_survives() -> None:
    assert "2.50" not in _text(), "the superseded $2.50 cap is 5.5x the approved ceiling"


def test_no_multi_hour_deadline_survives() -> None:
    """At any plausible rate a multi-hour deadline exceeds a $0.45 cap."""
    hits = re.findall(r"\b\d+\s*h\s*\d*\s*m?\b", _text())
    assert not hits, f"multi-hour deadline(s) left in the guide: {hits}"


def test_the_launch_example_uses_the_approved_cap() -> None:
    assert f"--usd-cap {APPROVED_CAP}" in _text()


def test_calibration_is_excluded_in_writing() -> None:
    assert "Calibration is NOT approved" in _text()


def test_the_deadline_rule_is_stated_generally_not_only_as_an_example() -> None:
    """A worked example at one rate is not a rule; the guide must state the rule."""
    assert "$0.45 / actual_rate" in _text()
