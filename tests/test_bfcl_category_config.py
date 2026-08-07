from pathlib import Path

import pytest

from eval.bfcl_category_config import (
    SCORING_CATEGORIES,
    SUPPORTED_CATEGORIES,
    resolve_category_paths,
)


def test_multiple_resolves_to_its_own_inputs_and_output() -> None:
    paths = resolve_category_paths(Path("/repo"), "multiple")
    assert paths.questions == Path("/repo/eval/bfcl_data/BFCL_v4_multiple.json")
    assert paths.answer_key == Path(
        "/repo/eval/bfcl_data/possible_answer/BFCL_v4_multiple.json"
    )
    assert paths.default_output == Path("/repo/eval/out/bfcl_multiple")


def test_simple_python_preserves_the_study1_output_path() -> None:
    paths = resolve_category_paths(Path("/repo"), "simple_python")
    assert paths.default_output == Path("/repo/eval/out/bfcl_simple")


def test_unknown_category_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported BFCL category"):
        resolve_category_paths(Path("/repo"), "irrelevance")


def test_live_multiple_is_the_development_category_and_carries_its_flag() -> None:
    """The flag is what activates the subset gate in bfcl_simple.main(), so it is
    pinned by a test rather than left to inspection."""
    paths = resolve_category_paths(Path("/repo"), "live_multiple")
    assert paths.questions == Path("/repo/eval/bfcl_data/BFCL_v4_live_multiple.json")
    assert paths.answer_key == Path(
        "/repo/eval/bfcl_data/possible_answer/BFCL_v4_live_multiple.json"
    )
    assert paths.default_output == Path("/repo/eval/out/bfcl_live_multiple")
    assert paths.is_development_subset is True


def test_the_reporting_categories_are_not_development_subsets() -> None:
    for category in ("simple_python", "multiple"):
        assert resolve_category_paths(Path("/repo"), category).is_development_subset is False


def test_live_multiple_is_supported_but_never_a_scoring_endpoint() -> None:
    assert "live_multiple" in SUPPORTED_CATEGORIES
    assert "live_multiple" not in SCORING_CATEGORIES
    assert set(SCORING_CATEGORIES) == {"simple_python", "multiple"}
