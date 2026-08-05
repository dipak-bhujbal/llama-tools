"""Tests for the answer-key preflight (prereg amendment 2 draft, §A2.3).

Exact function-name matching is only a fair rule while the key's expected name
is among the tools the item actually presented. Where it is not, the item is
unpassable by construction and the model is marked wrong for the benchmark's
defect — which is what `simple_python_363` is. `parsed_name == gt_name` cannot
notice that on its own, so the rule is two parts and this covers the second.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.bfcl_scoring import KeyDefectError, preflight_key_names

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "eval" / "bfcl_data"


def _question(item_id: str, *names: str) -> dict:
    return {"id": item_id, "function": [{"name": name} for name in names]}


def _answer(item_id: str, *names: str) -> dict:
    return {"id": item_id, "ground_truth": [{name: {"x": ["1"]}} for name in names]}


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


def test_a_key_naming_only_presented_tools_passes() -> None:
    questions = _by_id([_question("a", "pkg.fn"), _question("b", "other")])
    answers = _by_id([_answer("a", "pkg.fn"), _answer("b", "other")])

    assert preflight_key_names(questions, answers) == 2


def test_a_key_expecting_an_unpresented_name_fails_closed() -> None:
    """The simple_python_363 shape: key wants the unqualified tail, the item
    only ever offered the module-qualified name."""
    questions = _by_id([_question("a", "restaurant_search.find_closest")])
    answers = _by_id([_answer("a", "find_closest")])

    with pytest.raises(KeyDefectError, match="never presented to the model"):
        preflight_key_names(questions, answers)


def test_the_defect_message_names_the_offending_item() -> None:
    """A defect that does not say which item it is cannot be filed upstream."""
    questions = _by_id([_question("good", "fn"), _question("bad", "pkg.fn")])
    answers = _by_id([_answer("good", "fn"), _answer("bad", "fn")])

    with pytest.raises(KeyDefectError) as excinfo:
        preflight_key_names(questions, answers)

    message = str(excinfo.value)
    assert "bad: key expects ['fn']" in message
    assert "presented tools are ['pkg.fn']" in message
    assert "good" not in message


def test_an_answer_row_with_no_question_fails_closed() -> None:
    questions = _by_id([_question("a", "fn")])
    answers = _by_id([_answer("a", "fn"), _answer("orphan", "fn")])

    with pytest.raises(KeyDefectError, match="not in the questions file"):
        preflight_key_names(questions, answers)


def test_multi_name_keys_require_every_name_to_be_presented() -> None:
    questions = _by_id([_question("a", "fn_one")])
    answers = _by_id([_answer("a", "fn_one", "fn_two")])

    with pytest.raises(KeyDefectError, match=r"key expects \['fn_two'\]"):
        preflight_key_names(questions, answers)


def test_tail_matching_is_not_accepted_as_presented() -> None:
    """A tail-matching rule would pass this; exact matching must not."""
    questions = _by_id([_question("a", "circle_properties.get", "triangle_properties.get")])
    answers = _by_id([_answer("a", "get")])

    with pytest.raises(KeyDefectError, match="never presented"):
        preflight_key_names(questions, answers)


@pytest.mark.parametrize("category", ["simple_python", "multiple", "live_simple"])
@pytest.mark.skipif(
    not (DATA / "possible_answer").is_dir(),
    reason="pinned BFCL data is gitignored; run `python eval/fetch_pinned_bfcl.py` first",
)
def test_every_pinned_category_passes_the_preflight_today(category: str) -> None:
    """The measured claim §A2.3 rests on, checked rather than quoted: across
    the pinned categories, no key expects a name the item did not offer."""

    def load(path: Path) -> dict[str, dict]:
        return _by_id(
            [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        )

    questions = load(DATA / f"BFCL_v4_{category}.json")
    answers = load(DATA / "possible_answer" / f"BFCL_v4_{category}.json")

    assert preflight_key_names(questions, answers) == len(answers)
