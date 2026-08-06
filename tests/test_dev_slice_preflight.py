"""Fail-closed preflight for study 2's Decision-C development set.

The development set is a deterministic 258-item subset of the pinned
`live_multiple` parent. These tests bind the input bytes, reproduce the seeded
subset, enforce removal of the one final-set question collision, prove the
mining pool was re-screened against the parent, and keep the whole parent
category machine-labelled as development-only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from eval.bfcl_scoring import preflight_key_names
from eval.fetch_pinned_bfcl import verify_payload
from mining.dev_subset import build as build_dev_subset

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "eval" / "bfcl_data"
MANIFEST = REPO_ROOT / "eval" / "manifests" / "bfcl_v4_study2.json"
DECON_RECEIPT = (
    REPO_ROOT
    / "mining"
    / "receipts"
    / "sft_dedup_v2_decontamination_with_live_multiple.json"
)
SUBSET_RECEIPT = REPO_ROOT / "mining" / "receipts" / "study2_dev_look_subset.json"

DEV_CATEGORY = "live_multiple"
DEV_ROLE = "development_selection_only"
FINAL_CATEGORIES = {"multiple": 200, "simple_python": 400}

PARENT_ROW_COUNT = 1053
ELIGIBLE_ROW_COUNT = 1052
DEV_ROW_COUNT = 258
DEV_QUESTIONS_SHA256 = "fd8ccfad4d911420d0e3341dbe2fff77d1d341da934248b9bb2bda24ab3a10c8"
DEV_ANSWER_KEY_SHA256 = "97e90d59c5bd76c55a2920ce93e5566e9046307d3f558578f085f9d3a56c3084"
PARENT_SORTED_ID_SHA256 = "96d9015b2f01ea9a9a090afa8bd8638d81dccccd07d6632379dfc79a35c213ae"
DEV_SORTED_ID_SHA256 = "a91d8271224d7a50f68c27c0070b114173412c2591ba304ac7a6048506760b64"
MANIFEST_SHA256 = "542d407d434655487daa3faa0da69666cc5e5fa47c8ff67ab9771acc512fe3a0"
DECON_RECEIPT_SHA256 = "3daaffa85a2097468f53845d1cddf996a0e68a3605916e26918891c2972732b3"
SUBSET_RECEIPT_SHA256 = "5a9510711adee429b8d0b2d7e20b35cb57278d052f39cb19d33f86a46b57b33b"

OVERLAP_EXCLUSION = {
    "id": "live_multiple_190-84-0",
    "reason": "question_collision_with_final_set",
    "collides_with": "multiple:multiple_26",
}


def _rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _questions(category: str) -> list[dict]:
    return _rows(DATA / f"BFCL_v4_{category}.json")


def _answers(category: str) -> list[dict]:
    return _rows(DATA / "possible_answer" / f"BFCL_v4_{category}.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_question(row: dict) -> str:
    return json.dumps(row["question"], sort_keys=True, separators=(",", ":"))


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _manifest_entry(category: str, role: str) -> dict:
    matches = [
        entry
        for entry in _manifest()["files"]
        if entry["category"] == category and entry["role"] == role
    ]
    assert len(matches) == 1, f"expected exactly one {category}/{role} manifest entry"
    return matches[0]


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


def test_decision_c_artifacts_match_their_preregistered_digests() -> None:
    assert _sha256(MANIFEST) == MANIFEST_SHA256
    assert _sha256(DECON_RECEIPT) == DECON_RECEIPT_SHA256
    assert _sha256(SUBSET_RECEIPT) == SUBSET_RECEIPT_SHA256


def test_final_scoring_sets_have_the_roles_and_counts_the_analysis_assumes() -> None:
    categories = _manifest()["categories"]
    assert {
        category
        for category, spec in categories.items()
        if spec.get("study2_role") == "final_scoring"
    } == set(FINAL_CATEGORIES)

    for category, expected_rows in FINAL_CATEGORIES.items():
        entry = _manifest_entry(category, "questions")
        path = REPO_ROOT / entry["local_path"]
        assert entry["row_count"] == expected_rows
        assert entry["unique_id_count"] == expected_rows
        assert len(_rows(path)) == expected_rows
        verify_payload(path.read_bytes(), entry)


def test_live_multiple_parent_files_match_their_manifest_pins() -> None:
    for role, expected_sha in (
        ("questions", DEV_QUESTIONS_SHA256),
        ("answer_key", DEV_ANSWER_KEY_SHA256),
    ):
        entry = _manifest_entry(DEV_CATEGORY, role)
        path = REPO_ROOT / entry["local_path"]
        assert entry["sha256"] == expected_sha
        assert _sha256(path) == expected_sha
        assert entry["row_count"] == PARENT_ROW_COUNT
        assert entry["unique_id_count"] == PARENT_ROW_COUNT
        assert entry["sorted_id_sha256"] == PARENT_SORTED_ID_SHA256
        verify_payload(path.read_bytes(), entry)


def test_live_multiple_parent_is_key_valid_and_can_measure_tool_ranking() -> None:
    questions = _questions(DEV_CATEGORY)
    answers = _answers(DEV_CATEGORY)
    question_by_id = _by_id(questions)
    answer_by_id = _by_id(answers)

    assert len(question_by_id) == len(answer_by_id) == PARENT_ROW_COUNT
    assert set(question_by_id) == set(answer_by_id)
    assert {len(row["ground_truth"]) for row in answers} == {1}
    assert min(len(row["function"]) for row in questions) == 2
    assert max(len(row["function"]) for row in questions) == 37
    assert preflight_key_names(question_by_id, answer_by_id) == PARENT_ROW_COUNT


def test_seeded_subset_receipt_regenerates_exactly() -> None:
    committed = json.loads(SUBSET_RECEIPT.read_text())
    assert build_dev_subset(MANIFEST) == committed
    assert committed["criterion_id"] == "study2-dev-look-subset/v1"
    assert committed["seed"] == "study2-dev-look-subset/v1:20260806"
    assert committed["source"]["rows"] == PARENT_ROW_COUNT
    assert committed["eligible_rows"] == ELIGIBLE_ROW_COUNT
    assert committed["subset_size"] == DEV_ROW_COUNT
    assert len(committed["selected_ids"]) == DEV_ROW_COUNT
    assert len(set(committed["selected_ids"])) == DEV_ROW_COUNT
    assert committed["sorted_id_sha256"] == DEV_SORTED_ID_SHA256


def test_the_only_final_question_overlap_is_excluded_before_sampling() -> None:
    receipt = json.loads(SUBSET_RECEIPT.read_text())
    assert receipt["exclusions"] == [OVERLAP_EXCLUSION]
    assert OVERLAP_EXCLUSION["id"] not in receipt["selected_ids"]

    parent = _questions(DEV_CATEGORY)
    parent_questions = {_canonical_question(row): row["id"] for row in parent}
    parent_ids = set(_by_id(parent))
    selected_questions = {
        _canonical_question(row)
        for row in parent
        if row["id"] in set(receipt["selected_ids"])
    }

    overlaps: dict[str, list[tuple[str, str]]] = {}
    for category in FINAL_CATEGORIES:
        final = _questions(category)
        final_questions = {_canonical_question(row): row["id"] for row in final}
        overlaps[category] = sorted(
            (parent_questions[question], final_questions[question])
            for question in parent_questions.keys() & final_questions.keys()
        )
        assert not (parent_ids & set(_by_id(final)))
        assert not (selected_questions & final_questions.keys())

    assert overlaps == {
        "multiple": [("live_multiple_190-84-0", "multiple_26")],
        "simple_python": [],
    }


def test_re_screened_artifact_includes_the_entire_live_multiple_parent() -> None:
    receipt = json.loads(DECON_RECEIPT.read_text())
    screened = {
        entry["category"]: entry["sha256"]
        for entry in receipt["screened_question_files"]
    }

    assert receipt["manifest"]["sha256"] == MANIFEST_SHA256
    assert screened[DEV_CATEGORY] == DEV_QUESTIONS_SHA256
    assert receipt["dropped"] == {
        "multi": 903,
        "single": 108,
        "total": 1011,
        "by_reason": {"fn_name": 1010, "ngram_overlap": 1},
    }
    assert receipt["weights"] == {"n_multi": 8081, "n_single": 2990, "N": 11071}


def test_live_multiple_is_machine_disqualified_from_endpoint_reporting() -> None:
    manifest_role = _manifest()["categories"][DEV_CATEGORY]["study2_role"]
    subset = json.loads(SUBSET_RECEIPT.read_text())

    assert manifest_role == subset["study2_role"] == DEV_ROLE
    assert DEV_CATEGORY not in FINAL_CATEGORIES
    assert "never be reported as a study-2 endpoint" in subset["endpoint_status"]
    assert "category as a whole is spent" in subset["endpoint_status"]
