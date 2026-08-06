from pathlib import Path

import pytest

from eval.bfcl_category_config import resolve_category_paths


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
