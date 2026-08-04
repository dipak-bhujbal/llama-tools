"""Tests for the mined-prompt-vs-eval decontamination screen.

Covers the two independent signals (13-gram text overlap, exact function-name
overlap), the fail-closed sha256 check on the pinned eval files, and the
`screened_manifest()` audit trail that feeds the mining ledger. One test uses
the real pinned BFCL files under `eval/bfcl_data/` end-to-end so the screen is
proven against the actual eval set, not just synthetic fixtures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mining.decontaminate import Decontaminator, EvalIntegrityError

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MANIFEST = REPO_ROOT / "eval" / "manifests" / "bfcl_v4_study2.json"

# A verbatim question + function name from the real, pinned eval set
# (eval/bfcl_data/BFCL_v4_simple_python.json, id "simple_python_0").
REAL_EVAL_PROMPT = (
    "Find the area of a triangle with a base of 10 units and height of 5 units."
)
REAL_EVAL_FUNCTION = "calculate_triangle_area"


def _write_question_file(path: Path, rows: list[dict]) -> bytes:
    payload = ("\n".join(json.dumps(row) for row in rows) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return payload


def _write_manifest(manifest_path: Path, files: list[dict]) -> None:
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "files": files}, indent=2)
    )


@pytest.fixture
def synthetic_eval(tmp_path: Path) -> Path:
    """Build a small, self-contained eval set + manifest under tmp_path.

    One "questions" file (category "widgets") and one "answer_key" file
    (which must never be screened) that, if screened, would introduce a
    detectable extra function name ("answer_key_only_fn").
    """
    data_dir = tmp_path / "bfcl_data"
    data_dir.mkdir()

    questions_path = data_dir / "BFCL_widgets.json"
    questions_payload = _write_question_file(
        questions_path,
        [
            {
                "id": "widgets_0",
                "question": [
                    [
                        {
                            "role": "user",
                            "content": (
                                "Please schedule a meeting with the sales team "
                                "for next Tuesday at 10 AM in conference room B."
                            ),
                        }
                    ]
                ],
                "function": [{"name": "schedule_meeting", "description": "...", "parameters": {}}],
            }
        ],
    )

    answer_key_path = data_dir / "possible_answer" / "BFCL_widgets.json"
    answer_key_path.parent.mkdir(parents=True)
    answer_key_payload = _write_question_file(
        answer_key_path,
        [
            {
                "id": "widgets_0",
                "question": [[{"role": "user", "content": "irrelevant answer-key text"}]],
                "function": [
                    {"name": "answer_key_only_fn", "description": "...", "parameters": {}}
                ],
            }
        ],
    )

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "local_path": str(questions_path),
                "category": "widgets",
                "role": "questions",
                "sha256": hashlib.sha256(questions_payload).hexdigest(),
            },
            {
                "local_path": str(answer_key_path),
                "category": "widgets",
                "role": "answer_key",
                "sha256": hashlib.sha256(answer_key_payload).hexdigest(),
            },
        ],
    )
    return manifest_path


# ---------------------------------------------------------------------------
# n-gram screen
# ---------------------------------------------------------------------------


def test_verbatim_copy_is_flagged_with_ngram_reason(synthetic_eval: Path) -> None:
    d = Decontaminator([synthetic_eval], ngram=13)
    flagged, reason = d.is_contaminated(
        "Please schedule a meeting with the sales team for next Tuesday at 10 AM "
        "in conference room B.",
        function_names=[],
    )
    assert flagged is True
    assert reason == "ngram_overlap:widgets"


def test_case_and_whitespace_differences_still_match(synthetic_eval: Path) -> None:
    d = Decontaminator([synthetic_eval], ngram=13)
    mangled = (
        "  PLEASE   schedule a MEETING with the sales team for next tuesday   "
        "AT 10 am in conference ROOM b.  "
    )
    flagged, reason = d.is_contaminated(mangled, function_names=[])
    assert flagged is True
    assert reason == "ngram_overlap:widgets"


def test_unrelated_prompt_is_not_flagged(synthetic_eval: Path) -> None:
    d = Decontaminator([synthetic_eval], ngram=13)
    flagged, reason = d.is_contaminated(
        "What is the current weather forecast for Denver, Colorado this weekend?",
        function_names=["get_weather_forecast"],
    )
    assert flagged is False
    assert reason is None


def test_short_prompt_does_not_crash_and_is_not_ngram_flagged(synthetic_eval: Path) -> None:
    d = Decontaminator([synthetic_eval], ngram=13)
    # Fewer words than the eval question and far below ngram=13.
    flagged, reason = d.is_contaminated("schedule a meeting", function_names=[])
    assert flagged is False
    assert reason is None


def test_empty_prompt_does_not_crash(synthetic_eval: Path) -> None:
    d = Decontaminator([synthetic_eval], ngram=13)
    flagged, reason = d.is_contaminated("", function_names=[])
    assert flagged is False
    assert reason is None


# ---------------------------------------------------------------------------
# function-name screen
# ---------------------------------------------------------------------------


def test_shared_function_name_is_flagged_even_with_unrelated_prompt(
    synthetic_eval: Path,
) -> None:
    d = Decontaminator([synthetic_eval], ngram=13)
    flagged, reason = d.is_contaminated(
        "Convert 5 kilometers to miles for me please.",
        function_names=["schedule_meeting"],
    )
    assert flagged is True
    assert reason == "fn_name:schedule_meeting:widgets"


def test_unrelated_function_name_is_not_flagged(synthetic_eval: Path) -> None:
    d = Decontaminator([synthetic_eval], ngram=13)
    flagged, reason = d.is_contaminated(
        "Convert 5 kilometers to miles for me please.",
        function_names=["convert_distance"],
    )
    assert flagged is False
    assert reason is None


def test_answer_key_function_names_are_never_indexed(synthetic_eval: Path) -> None:
    """role == 'answer_key' must be skipped entirely (requirement 5)."""
    d = Decontaminator([synthetic_eval], ngram=13)
    flagged, reason = d.is_contaminated("anything at all", function_names=["answer_key_only_fn"])
    assert flagged is False
    assert reason is None


# ---------------------------------------------------------------------------
# screened_manifest()
# ---------------------------------------------------------------------------


def test_screened_manifest_includes_only_questions_role(synthetic_eval: Path) -> None:
    d = Decontaminator([synthetic_eval], ngram=13)
    d.is_contaminated("trigger the lazy load", function_names=[])

    manifest = d.screened_manifest()

    assert len(manifest) == 1
    entry = manifest[0]
    assert entry["role"] == "questions"
    assert entry["category"] == "widgets"
    assert "sha256" in entry and len(entry["sha256"]) == 64
    assert "local_path" in entry


def test_screened_manifest_alone_triggers_loading(synthetic_eval: Path) -> None:
    """screened_manifest() must work even if is_contaminated was never called."""
    d = Decontaminator([synthetic_eval], ngram=13)
    manifest = d.screened_manifest()
    assert len(manifest) == 1


# ---------------------------------------------------------------------------
# fail-closed sha256 check
# ---------------------------------------------------------------------------


def test_tampered_sha256_in_manifest_raises(tmp_path: Path) -> None:
    data_dir = tmp_path / "bfcl_data"
    data_dir.mkdir()
    questions_path = data_dir / "BFCL_widgets.json"
    _write_question_file(
        questions_path,
        [{"id": "w0", "question": [[{"role": "user", "content": "hello world"}]], "function": []}],
    )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "local_path": str(questions_path),
                "category": "widgets",
                "role": "questions",
                "sha256": "0" * 64,  # deliberately wrong
            }
        ],
    )

    d = Decontaminator([manifest_path], ngram=13)
    with pytest.raises(EvalIntegrityError, match="sha256"):
        d.is_contaminated("hello world", function_names=[])


def test_tampered_file_on_disk_after_manifest_pin_raises(tmp_path: Path) -> None:
    """Simulates the file being edited post-pin, not just a bad manifest value."""
    data_dir = tmp_path / "bfcl_data"
    data_dir.mkdir()
    questions_path = data_dir / "BFCL_widgets.json"
    original_payload = _write_question_file(
        questions_path,
        [{"id": "w0", "question": [[{"role": "user", "content": "hello world"}]], "function": []}],
    )
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "local_path": str(questions_path),
                "category": "widgets",
                "role": "questions",
                "sha256": hashlib.sha256(original_payload).hexdigest(),
            }
        ],
    )

    # Mutate the file after it was pinned.
    questions_path.write_bytes(original_payload + b'{"id": "w1", "question": [], "function": []}\n')

    d = Decontaminator([manifest_path], ngram=13)
    with pytest.raises(EvalIntegrityError):
        d.screened_manifest()


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------


def test_loading_is_cached_after_first_use(synthetic_eval: Path) -> None:
    """After the first call, later calls must not touch disk again.

    Deletes the manifest (and implicitly stops any re-read of the eval data)
    right after the first call. If loading were not cached, the next call
    would raise FileNotFoundError instead of returning a normal verdict.
    """
    d = Decontaminator([synthetic_eval], ngram=13)
    d.is_contaminated("trigger the lazy load", function_names=[])

    synthetic_eval.unlink()

    flagged, reason = d.is_contaminated(
        "Please schedule a meeting with the sales team for next Tuesday at 10 AM "
        "in conference room B.",
        function_names=[],
    )
    assert flagged is True
    assert reason == "ngram_overlap:widgets"


def test_repeated_calls_return_consistent_results(synthetic_eval: Path) -> None:
    d = Decontaminator([synthetic_eval], ngram=13)
    results = [
        d.is_contaminated("Convert 5 kilometers to miles for me please.", function_names=[])
        for _ in range(10)
    ]
    assert all(r == (False, None) for r in results)


# ---------------------------------------------------------------------------
# End-to-end against the real, pinned BFCL v4 eval set
# ---------------------------------------------------------------------------


def test_real_manifest_flags_verbatim_eval_question_via_ngram() -> None:
    if not REAL_MANIFEST.exists():
        pytest.skip("real BFCL manifest not present in this checkout")

    d = Decontaminator([REAL_MANIFEST], ngram=13)
    flagged, reason = d.is_contaminated(REAL_EVAL_PROMPT, function_names=[])

    assert flagged is True
    assert reason == "ngram_overlap:simple_python"


def test_real_manifest_flags_shared_function_name() -> None:
    if not REAL_MANIFEST.exists():
        pytest.skip("real BFCL manifest not present in this checkout")

    d = Decontaminator([REAL_MANIFEST], ngram=13)
    flagged, reason = d.is_contaminated(
        "This prompt text shares nothing with any eval question at all.",
        function_names=[REAL_EVAL_FUNCTION],
    )

    assert flagged is True
    assert reason == f"fn_name:{REAL_EVAL_FUNCTION}:simple_python"


def test_real_manifest_unrelated_prompt_and_function_pass_clean() -> None:
    if not REAL_MANIFEST.exists():
        pytest.skip("real BFCL manifest not present in this checkout")

    d = Decontaminator([REAL_MANIFEST], ngram=13)
    flagged, reason = d.is_contaminated(
        "Compose a birthday poem for my grandmother who loves gardening and tea.",
        function_names=["compose_birthday_poem_for_gardener"],
    )

    assert flagged is False
    assert reason is None


def test_real_manifest_screened_manifest_matches_committed_manifest_questions() -> None:
    if not REAL_MANIFEST.exists():
        pytest.skip("real BFCL manifest not present in this checkout")

    d = Decontaminator([REAL_MANIFEST], ngram=13)
    screened = d.screened_manifest()

    committed = json.loads(REAL_MANIFEST.read_text())
    expected_question_categories = {
        spec["category"] for spec in committed["files"] if spec["role"] == "questions"
    }

    assert {entry["category"] for entry in screened} == expected_question_categories
    assert all(entry["role"] == "questions" for entry in screened)
