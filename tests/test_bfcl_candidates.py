"""Candidate-list assembly for the BFCL sweep.

These cover the `base` candidate added for the study-2 qualification probe,
which needs base-vs-SFT lift numbers. `base` is not an adapter — it is the
unmodified base model generated with adapters disabled — so the invariant that
matters is that it never reaches `set_adapter()`.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from bfcl_simple import build_candidate_names


def test_sft_only_default_has_no_base():
    """Study-1 behaviour is unchanged when --include-base is absent."""
    assert build_candidate_names(include_base=False, sft_only=True, checkpoints=[]) == ["sft"]


def test_include_base_prepends_base():
    assert build_candidate_names(include_base=True, sft_only=True, checkpoints=[]) == [
        "base",
        "sft",
    ]


def test_base_is_first_so_the_baseline_is_scored_before_adapters():
    names = build_candidate_names(include_base=True, sft_only=False, checkpoints=[50, 100])
    assert names[0] == "base"
    assert names == ["base", "sft", "dpo-50", "dpo-100"]


def test_checkpoints_are_skipped_when_sft_only():
    names = build_candidate_names(include_base=True, sft_only=True, checkpoints=[50, 100])
    assert names == ["base", "sft"]


def test_sft_is_always_present():
    """Every sweep must score the shipped baseline it is compared against."""
    for include_base in (True, False):
        for sft_only in (True, False):
            names = build_candidate_names(
                include_base=include_base, sft_only=sft_only, checkpoints=[50]
            )
            assert "sft" in names


@pytest.mark.parametrize("checkpoints", [None, []])
def test_missing_checkpoints_do_not_crash(checkpoints):
    assert build_candidate_names(
        include_base=False, sft_only=False, checkpoints=checkpoints
    ) == ["sft"]


def test_base_appears_at_most_once():
    names = build_candidate_names(include_base=True, sft_only=False, checkpoints=[50, 100, 150])
    assert names.count("base") == 1


def test_no_dpo_candidate_is_named_base():
    """Guards the set_adapter() invariant: only 'base' means adapters-disabled."""
    names = build_candidate_names(include_base=True, sft_only=False, checkpoints=[50, 100])
    adapter_names = [n for n in names if n != "base"]
    assert all(n == "sft" or n.startswith("dpo-") for n in adapter_names)
