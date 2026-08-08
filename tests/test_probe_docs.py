"""The probe operator guide is load-bearing, so its *pasteable* commands are tested.

The guide once carried a superseded cap in its launch example for hours after a
smaller one had been approved, and `launch_probe.sh` derived its wall-clock
`timeout` from that flag. A stale figure in a pasteable command therefore did not
produce an *unenforced* budget; it produced a **mechanically enforced one at a
number nobody had approved**, with everything downstream looking correct.

The fix was structural: money left the executable source, so the only remaining
way for a stale figure to be enforced is for someone to paste one out of this
guide. These tests police exactly that boundary — **prose may record the approval
as audit/policy; a fenced command block may not carry a currency amount.**

Stated as an invariant and deliberately without naming any amount: repeating the
superseded or the current figure here would recreate, in the test suite, the same
duplicate-source-of-truth this change exists to remove.
"""

from __future__ import annotations

import re
from pathlib import Path

GUIDE = Path("docs/probe-bootstrap.md")

_FENCE_RE = re.compile(r"^```.*?^```", re.M | re.S)
# A currency amount is a `$` carrying a decimal (`$0.45`), a `$` next to a rate
# word (`$45/hr`), or a bare decimal next to one (`0.44usd/hr`). Bash's `$1` in
# `awk '{print $1}'` and `${VAR}` are positional parameters, not money.
_MONEY_RE = re.compile(
    r"\$\s*\d+\.\d+"
    r"|\$\s*\d+\s*(?:/\s*hr|usd|USD|per[- ]hour)"
    r"|\d+\.\d+\s*(?:usd|USD|/\s*hr|per[- ]hour)"
)


def _text() -> str:
    return GUIDE.read_text()


def _code_blocks() -> list[str]:
    return _FENCE_RE.findall(_text())


def test_the_guide_has_pasteable_command_blocks_to_police() -> None:
    """Guard the guard: the other tests are vacuous if the fences stop matching."""
    blocks = _code_blocks()
    assert len(blocks) >= 3, blocks
    assert any("launch_probe.sh" in b for b in blocks), "no launch example found"


def test_no_currency_amount_appears_in_any_pasteable_command() -> None:
    for block in _code_blocks():
        hits = _MONEY_RE.findall(block)
        assert not hits, f"currency amount in a pasteable command: {hits}\n{block}"


def test_the_launch_example_passes_max_seconds_and_no_money_flags() -> None:
    launch = [b for b in _code_blocks() if "launch_probe.sh" in b]
    assert launch, "no launch example found"
    for block in launch:
        assert "--max-seconds" in block
        assert "--usd-cap" not in block
        assert "--usd-per-hour" not in block


def test_no_multi_hour_deadline_survives() -> None:
    """At any plausible pod rate a multi-hour deadline blows the approved ceiling."""
    hits = re.findall(r"\b\d+\s*h\s*\d*\s*m?\b", _text())
    assert not hits, f"multi-hour deadline(s) left in the guide: {hits}"


def test_the_derivation_is_from_time_remaining_not_the_full_approved_ceiling() -> None:
    """Bootstrap is billed time. Re-deriving from the full ceiling grants it twice.

    Cloning, venv creation and a 16 GB weight download all run on the meter, so
    by launch some of the approved ceiling is already spent. The bound the
    script gets must come from what is left until the provider deadline.
    """
    assert "provider_termination_epoch" in _text()


def test_the_two_bounds_are_documented_as_independent() -> None:
    """The script's timeout dies with the script; only the provider's outlives it."""
    text = _text()
    assert "independent bounds" in text
    assert "auto-termination" in text


def test_calibration_is_excluded_in_writing() -> None:
    assert "Calibration is NOT approved" in _text()
