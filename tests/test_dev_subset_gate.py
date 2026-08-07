"""Tests for the development-subset gate (prereg §3.2, §3.3).

Every kill line and eligibility comparison in §3 is denominated in *items* against
the frozen baseline's 258. So the failure that matters is not "the gate crashed" —
it is "the gate quietly scored 257 items, or 258 slightly different ones", which
would move every threshold that referenced the baseline without anything looking
wrong. Each test below is one way that could happen, and each asserts a refusal.

The happy-path test uses the real committed receipt. The failure tests build a
synthetic mini-repository in `tmp_path` so they can corrupt one thing at a time
without touching a committed artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval import dev_subset_gate
from eval.dev_subset_gate import DevSubsetGateError, load_dev_subset_ids

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_real_committed_subset_loads_and_is_exactly_the_pinned_set() -> None:
    ids = load_dev_subset_ids(REPO_ROOT)
    receipt = json.loads(
        (REPO_ROOT / "mining" / "receipts" / "study2_dev_look_subset.json").read_text()
    )

    assert len(ids) == 258
    assert len(set(ids)) == 258
    assert ids == receipt["selected_ids"]


def _mini_repo(tmp_path: Path, *, ids: list[str], role: str = "development_selection_only",
               manifest_digest_override: str | None = None) -> Path:
    """A synthetic repo whose digests are internally consistent unless told otherwise."""
    questions = tmp_path / "eval" / "bfcl_data" / "BFCL_v4_live_multiple.json"
    answer_key = tmp_path / "eval" / "bfcl_data" / "possible_answer" / "BFCL_v4_live_multiple.json"
    for path in (questions, answer_key):
        path.parent.mkdir(parents=True, exist_ok=True)

    rows = [{"id": i, "question": [], "function": [{"name": "f"}, {"name": "g"}]} for i in ids]
    questions.write_text("".join(json.dumps(r) + "\n" for r in rows))
    answer_key.write_text("".join(json.dumps({"id": i, "ground_truth": [{}]}) + "\n" for i in ids))

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "categories": {"live_multiple": {"study2_role": role}},
        "files": [
            {"category": "live_multiple", "role": "questions",
             "local_path": "eval/bfcl_data/BFCL_v4_live_multiple.json", "sha256": sha(questions)},
            {"category": "live_multiple", "role": "answer_key",
             "local_path": "eval/bfcl_data/possible_answer/BFCL_v4_live_multiple.json",
             "sha256": sha(answer_key)},
        ],
    }
    manifest_path = tmp_path / "eval" / "manifests" / "bfcl_v4_study2.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    receipt = {
        "manifest": {"sha256": manifest_digest_override or sha(manifest_path)},
        "selected_ids": ids,
    }
    receipt_path = tmp_path / "mining" / "receipts" / "study2_dev_look_subset.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return tmp_path


def _pin(monkeypatch, repo: Path, *, ids: list[str]) -> None:
    """Point the module's pins at this synthetic repo, so only the tested defect differs."""
    receipt_path = repo / "mining" / "receipts" / "study2_dev_look_subset.json"
    monkeypatch.setattr(
        dev_subset_gate, "SUBSET_RECEIPT_SHA256",
        hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        dev_subset_gate, "SUBSET_SORTED_ID_SHA256",
        hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest(),
    )
    monkeypatch.setattr(dev_subset_gate, "DEV_SUBSET_SIZE", len(ids))


def test_a_missing_receipt_refuses_rather_than_scoring_the_whole_parent(tmp_path) -> None:
    with pytest.raises(DevSubsetGateError, match="missing"):
        load_dev_subset_ids(tmp_path)


def test_a_receipt_whose_bytes_moved_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(dev_subset_gate, "SUBSET_RECEIPT_SHA256", "0" * 64)

    with pytest.raises(DevSubsetGateError, match="receipt sha256"):
        load_dev_subset_ids(REPO_ROOT)


def test_a_receipt_built_against_a_different_manifest_is_refused(tmp_path, monkeypatch) -> None:
    """The subset is only meaningful relative to the manifest it was drawn from."""
    ids = [f"live_multiple_{i}" for i in range(5)]
    repo = _mini_repo(tmp_path, ids=ids, manifest_digest_override="f" * 64)
    _pin(monkeypatch, repo, ids=ids)

    with pytest.raises(DevSubsetGateError, match="built against manifest"):
        load_dev_subset_ids(repo)


def test_a_category_no_longer_labelled_development_only_is_refused(tmp_path, monkeypatch) -> None:
    """If the role ever flips, the set has become reportable and must not be
    used for selection without that being a decision someone made."""
    ids = [f"live_multiple_{i}" for i in range(5)]
    repo = _mini_repo(tmp_path, ids=ids, role="final_scoring")
    _pin(monkeypatch, repo, ids=ids)

    with pytest.raises(DevSubsetGateError, match="not 'development_selection_only'"):
        load_dev_subset_ids(repo)


def test_a_tampered_question_file_is_refused(tmp_path, monkeypatch) -> None:
    ids = [f"live_multiple_{i}" for i in range(5)]
    repo = _mini_repo(tmp_path, ids=ids)
    _pin(monkeypatch, repo, ids=ids)
    questions = repo / "eval" / "bfcl_data" / "BFCL_v4_live_multiple.json"
    questions.write_text(questions.read_text() + json.dumps({"id": "extra"}) + "\n")

    with pytest.raises(DevSubsetGateError, match="sha256"):
        load_dev_subset_ids(repo)


def test_the_wrong_number_of_ids_is_refused(tmp_path, monkeypatch) -> None:
    """257 items is the failure this gate exists for: it looks like a run."""
    ids = [f"live_multiple_{i}" for i in range(5)]
    repo = _mini_repo(tmp_path, ids=ids)
    _pin(monkeypatch, repo, ids=ids)
    monkeypatch.setattr(dev_subset_gate, "DEV_SUBSET_SIZE", 6)

    with pytest.raises(DevSubsetGateError, match="expected 6 unique"):
        load_dev_subset_ids(repo)


def test_a_different_258_is_refused_by_the_sorted_id_digest(tmp_path, monkeypatch) -> None:
    """Right count, wrong items — the defect a size check alone cannot catch."""
    ids = [f"live_multiple_{i}" for i in range(5)]
    repo = _mini_repo(tmp_path, ids=ids)
    _pin(monkeypatch, repo, ids=ids)
    monkeypatch.setattr(dev_subset_gate, "SUBSET_SORTED_ID_SHA256", "a" * 64)

    with pytest.raises(DevSubsetGateError, match="sorted-ID digest"):
        load_dev_subset_ids(repo)


def test_an_id_absent_from_the_question_file_is_refused(tmp_path, monkeypatch) -> None:
    ids = [f"live_multiple_{i}" for i in range(5)]
    repo = _mini_repo(tmp_path, ids=ids)
    # Receipt asks for an id the questions file does not contain.
    receipt_path = repo / "mining" / "receipts" / "study2_dev_look_subset.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["selected_ids"] = [*ids[:-1], "live_multiple_absent"]
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    _pin(monkeypatch, repo, ids=receipt["selected_ids"])

    with pytest.raises(DevSubsetGateError, match="absent from"):
        load_dev_subset_ids(repo)


# --- The runner's restriction of *both* files (prereg §3.2) -------------------
#
# The gate above proves the right 258 ids are loaded. These prove the runner
# then scores exactly those 258 on both sides. That distinction is not academic:
# the first version of this scorer filtered only the questions and left the
# answer key at all 1,053 parent rows, so every development run would have died
# on id reconciliation — the one path the gate's own tests never exercised.


def _rows(ids: list[str]) -> list[dict]:
    return [{"id": i, "ground_truth": [{"f": {}}]} for i in ids]


def test_both_sides_are_restricted_to_the_pinned_ids() -> None:
    pinned = [f"live_multiple_{i}" for i in range(3)]
    parent = [*pinned, "live_multiple_excluded", "live_multiple_99"]

    prompts, answers = dev_subset_gate.restrict_to_dev_subset(
        _rows(parent), _rows(parent), pinned
    )

    assert [r["id"] for r in prompts] == pinned
    assert [r["id"] for r in answers] == pinned, (
        "the answer key must be cut to the same 258, not left at parent size"
    )


def test_a_pinned_id_with_no_answer_row_is_refused() -> None:
    pinned = [f"live_multiple_{i}" for i in range(3)]

    with pytest.raises(DevSubsetGateError, match="no answer row"):
        dev_subset_gate.restrict_to_dev_subset(
            _rows(pinned), _rows(pinned[:-1]), pinned
        )


def test_a_pinned_id_with_no_question_row_is_refused() -> None:
    pinned = [f"live_multiple_{i}" for i in range(3)]

    with pytest.raises(DevSubsetGateError, match="no question row"):
        dev_subset_gate.restrict_to_dev_subset(
            _rows(pinned[:-1]), _rows(pinned), pinned
        )


def test_a_duplicated_answer_row_is_refused() -> None:
    pinned = [f"live_multiple_{i}" for i in range(3)]

    with pytest.raises(DevSubsetGateError, match="duplicate pinned ids"):
        dev_subset_gate.restrict_to_dev_subset(
            _rows(pinned), _rows([*pinned, pinned[0]]), pinned
        )


def test_a_duplicated_pin_is_refused() -> None:
    pinned = ["live_multiple_0", "live_multiple_1", "live_multiple_0"]

    with pytest.raises(DevSubsetGateError, match="pinned ID list carries duplicates"):
        dev_subset_gate.restrict_to_dev_subset(_rows(pinned), _rows(pinned), pinned)


def test_the_real_258_restrict_cleanly_against_the_committed_parent_files() -> None:
    ids = load_dev_subset_ids(REPO_ROOT)
    data = REPO_ROOT / "eval" / "bfcl_data"

    def load(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    prompts, answers = dev_subset_gate.restrict_to_dev_subset(
        load(data / "BFCL_v4_live_multiple.json"),
        load(data / "possible_answer" / "BFCL_v4_live_multiple.json"),
        ids,
    )

    assert len(prompts) == len(answers) == 258
    assert {r["id"] for r in prompts} == {r["id"] for r in answers} == set(ids)
