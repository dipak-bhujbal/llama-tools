"""Tests for the synthetic DPO pair fixtures under `tests/fixtures/`.

Two jobs. First, reproduction: the committed JSONL files must be byte-identical
to a fresh run of `scale_pairs_fixed.py`, so the fixture set is an artifact with
a generator behind it rather than three opaque blobs that happened to arrive.
Second, Ground Rules 1-2: every row must be labeled synthetic with an honest
provenance string and carry no fabricated verifier metadata, so these pairs can
never be mistaken for mined evidence.

The structural assertions in the generator are a FIXTURE self-test. They are not
the verifier gate from HANDOFF.md 2.2 ("0 false positives, 0 misses"), which
belongs to `mining/mine_pairs.py` and is tested separately.
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
GENERATOR = FIXTURE_DIR / "scale_pairs_fixed.py"

# Metadata fields that would imply a real generation or verifier verdict.
# The legacy purged set carried these on synthetic rows; see HANDOFF.md 2.3.
FABRICATED_META_FIELDS = {
    "pass_rate",
    "verified_by",
    "gen_temperature",
    "source_dataset",
    "verifier_version",
}


def _load_generator():
    spec = importlib.util.spec_from_file_location("scale_pairs_fixed", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_generator()


def _rows(name: str) -> list[dict]:
    path = FIXTURE_DIR / name
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def splits() -> dict[str, list[dict]]:
    return {
        "train": _rows("fixture_pairs_train.jsonl"),
        "eval": _rows("fixture_pairs_eval.jsonl"),
        "audit": _rows("fixture_audit_sample_50.jsonl"),
    }


def test_committed_fixtures_reproduce_byte_for_byte(generator):
    """The committed bytes are what the pinned seed actually produces."""
    assert generator.check_fixtures(FIXTURE_DIR) == []


def test_generator_seed_is_pinned(generator):
    """Reproduction is meaningless if the seed can drift."""
    assert GENERATOR.read_text().count("random.Random(20260804)") == 1


def test_split_sizes(splits):
    assert len(splits["train"]) == 1440
    assert len(splits["eval"]) == 160
    assert len(splits["audit"]) == 50


def test_every_row_is_labeled_synthetic(splits):
    """Ground Rule 1: synthetic fixtures are fine only when labeled as such."""
    for name, rows in splits.items():
        for row in rows:
            assert row["meta"]["synthetic"] is True, f"{name}: {row['meta']['pair_id']}"


def test_provenance_is_honest_and_names_the_real_generator(splits):
    """The provenance string must name the file that actually produced the row."""
    for name, rows in splits.items():
        for row in rows:
            provenance = row["meta"]["provenance"]
            assert "tests/fixtures/scale_pairs_fixed.py" in provenance, name
            assert "NOT on-policy model generations" in provenance, name
            assert "not for training or publication as evidence" in provenance, name


def test_no_fabricated_verifier_metadata(splits):
    """Nothing may imply a generation or verdict that never happened."""
    for name, rows in splits.items():
        for row in rows:
            leaked = FABRICATED_META_FIELDS & set(row["meta"])
            assert not leaked, f"{name}: {row['meta']['pair_id']} carries {leaked}"


def test_no_purged_verifier_string_anywhere():
    """The Phase 0 acceptance grep, enforced as a test."""
    for path in FIXTURE_DIR.glob("*.jsonl"):
        assert "exact_match_checker_v2" not in path.read_text(), path.name


def test_error_type_allocation_matches_the_study_mix(generator, splits):
    combined = splits["train"] + splits["eval"]
    counts = Counter(row["meta"]["error_type"] for row in combined)
    assert dict(counts) == generator.ALLOC


def test_train_and_eval_are_disjoint(splits):
    train_ids = {row["meta"]["pair_id"] for row in splits["train"]}
    eval_ids = {row["meta"]["pair_id"] for row in splits["eval"]}
    assert not (train_ids & eval_ids)
    assert len(train_ids) + len(eval_ids) == generator_total()


def test_audit_sample_is_drawn_from_train(splits):
    train_ids = {row["meta"]["pair_id"] for row in splits["train"]}
    audit_ids = {row["meta"]["pair_id"] for row in splits["audit"]}
    assert audit_ids <= train_ids


def test_prompts_are_unique_across_the_whole_set(splits):
    combined = splits["train"] + splits["eval"]
    prompts = [row["prompt"][1]["content"] for row in combined]
    assert len(set(prompts)) == len(prompts)


def test_fixtures_are_ascii_only(splits):
    """Non-ASCII (em dashes especially) leaks into trainer logs and diffs."""
    for path in FIXTURE_DIR.glob("*.jsonl"):
        text = path.read_text()
        assert all(ord(ch) < 128 for ch in text), path.name


def test_dpo_schema_shape(splits):
    """TRL DPOTrainer expects prompt/chosen/rejected as message lists."""
    for name, rows in splits.items():
        for row in rows:
            assert [m["role"] for m in row["prompt"]] == ["system", "user"], name
            assert [m["role"] for m in row["chosen"]] == ["assistant"], name
            assert [m["role"] for m in row["rejected"]] == ["assistant"], name


def generator_total() -> int:
    return 1600
