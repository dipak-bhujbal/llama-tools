"""Preflight for the study-2 development set `D` (prereg §3.2, candidate).

`D` is the whole pinned `live_simple` category. Every property §3.2 claims about
it — its digests, its size, the structural blind spot that decides the selection
rule, and its disjointness from both final scoring sets — is checked here, so the
document cannot state one thing while the data says another.

The load-bearing one is `test_every_dev_item_presents_exactly_one_function`. §3.9
selects checkpoints by step count among healthy ones rather than by dev accuracy
*because* `D` cannot exercise ranking. If that ever stops being true, the reason
for the selection rule has changed and the rule has to be re-argued, not quietly
kept.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from eval.fetch_pinned_bfcl import verify_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "eval" / "bfcl_data"
MANIFEST = REPO_ROOT / "eval" / "manifests" / "bfcl_v4_study2.json"
DECON_RECEIPT = REPO_ROOT / "mining" / "receipts" / "sft_dedup_v2_decontamination.json"

DEV_CATEGORY = "live_simple"
FINAL_CATEGORIES = ("multiple", "simple_python")

# Quoted from prereg §3.2. Changing either side without the other is the defect
# this file exists to catch.
DEV_ROW_COUNT = 258
DEV_QUESTIONS_SHA256 = "1af2ac87dca47556db7b7e37e51e28b459a38b594e3c7b3c792b4903598ca0c4"
DEV_ANSWER_KEY_SHA256 = "fec9cfa9744a936f9126981e85a2023da1e63e273eafebc81923a1162fad70ce"
DEV_SORTED_ID_SHA256 = "aa668d6c39d5c7ca6080eced2e43a4573a30b506db7fa84a6d91bd7d6fd05ce3"
DISCLOSED_SHARED_FUNCTION_NAMES = {
    "multiple": {"send_email"},
    "simple_python": {"get_current_weather", "send_email"},
}


def _rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _questions(category: str) -> list[dict]:
    return _rows(DATA / f"BFCL_v4_{category}.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _presented_names(rows: list[dict]) -> set[str]:
    return {fn["name"] for row in rows for fn in row["function"]}


def _manifest_entry(category: str, role: str) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    matches = [
        entry
        for entry in manifest["files"]
        if entry["category"] == category and entry["role"] == role
    ]
    assert len(matches) == 1, f"expected exactly one {category}/{role} manifest entry"
    return matches[0]


def test_dev_files_match_their_manifest_pins() -> None:
    for role, expected_sha in (
        ("questions", DEV_QUESTIONS_SHA256),
        ("answer_key", DEV_ANSWER_KEY_SHA256),
    ):
        entry = _manifest_entry(DEV_CATEGORY, role)
        path = REPO_ROOT / entry["local_path"]

        assert entry["sha256"] == expected_sha, f"{role}: manifest pin moved off §3.2"
        assert _sha256(path) == expected_sha, f"{role}: file no longer matches its pin"
        assert entry["row_count"] == DEV_ROW_COUNT
        assert entry["unique_id_count"] == DEV_ROW_COUNT
        assert entry["sorted_id_sha256"] == DEV_SORTED_ID_SHA256


def test_dev_set_size_and_ids_are_what_the_prereg_says() -> None:
    """Verified through the fetcher's own checker, not a second recipe.

    The sorted-id digest has an exact serialization, and re-deriving it here
    would let this file agree with itself while disagreeing with the code that
    fetches the data.
    """
    rows = _questions(DEV_CATEGORY)
    ids = [row["id"] for row in rows]

    assert len(rows) == DEV_ROW_COUNT
    assert len(set(ids)) == DEV_ROW_COUNT

    for role in ("questions", "answer_key"):
        entry = _manifest_entry(DEV_CATEGORY, role)
        payload = (REPO_ROOT / entry["local_path"]).read_bytes()
        verify_payload(payload, entry)  # raises VerificationError on any mismatch


def test_every_dev_item_presents_exactly_one_function() -> None:
    """§3.2's structural blind spot, and therefore §3.9's selection rule."""
    counts = {len(row["function"]) for row in _questions(DEV_CATEGORY)}

    assert counts == {1}, (
        "live_simple no longer presents exactly one candidate function per item; "
        "prereg §3.2's blind-spot argument and §3.9's selection rule both depend "
        "on this and must be re-argued before an arm runs"
    )


def test_dev_set_is_disjoint_from_both_final_scoring_sets() -> None:
    dev = _questions(DEV_CATEGORY)
    dev_ids = {row["id"] for row in dev}
    dev_questions = {json.dumps(row["question"], sort_keys=True) for row in dev}

    for category in FINAL_CATEGORIES:
        final = _questions(category)
        final_ids = {row["id"] for row in final}
        final_questions = {json.dumps(row["question"], sort_keys=True) for row in final}

        assert not (dev_ids & final_ids), f"dev/{category} share item ids"
        assert not (dev_questions & final_questions), f"dev/{category} share a question"


def test_shared_function_names_are_exactly_the_disclosed_ones() -> None:
    """Names are disclosed in §3.2, not removed — but only these names."""
    dev_names = _presented_names(_questions(DEV_CATEGORY))

    for category, disclosed in DISCLOSED_SHARED_FUNCTION_NAMES.items():
        shared = dev_names & _presented_names(_questions(category))
        assert shared == disclosed, (
            f"dev/{category} function-name overlap changed: {sorted(shared)} "
            f"vs disclosed {sorted(disclosed)}"
        )


def test_dev_set_was_screened_by_the_frozen_decontamination_artifact() -> None:
    """§3.2's decisive argument for `D` over `live_multiple`."""
    receipt = json.loads(DECON_RECEIPT.read_text())
    screened = {
        entry["category"]: entry["sha256"]
        for entry in receipt["screened_question_files"]
    }

    assert DEV_CATEGORY in screened, (
        "the mining pool was not screened against the development set; "
        "selection would be run against a set the training data may contain"
    )
    assert screened[DEV_CATEGORY] == DEV_QUESTIONS_SHA256, (
        "the screened live_simple revision is not the one §3.2 pins"
    )
