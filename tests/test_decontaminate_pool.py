"""Tests for the decontamination artifact writer.

§2.5's weights are derived from this artifact and nothing else, so what matters
here is that the screen input is the *retained* population rather than the raw
file, that the weights leave as an exact integer triple, and that a row which
survives eligibility but cannot be screened stops the run instead of quietly
becoming a third bucket.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mining.decontaminate_pool import CRITERION_ID, DecontaminationError, build_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "eval" / "manifests" / "bfcl_v4_study2.json"


def _row(source_id: str, tools: list[str], user: str, target: str) -> dict:
    return {
        "source_id": source_id,
        "messages": [
            {"role": "system", "content": "Tools:\n" + json.dumps([{"name": t} for t in tools])},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
    }


def _pool(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "pool.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _preflight(tmp_path: Path, pool: Path, **overrides) -> Path:
    """A matching eligibility receipt. The producer refuses to screen without
    one that is about this pool and passed, so fixtures must supply it."""
    receipt = {
        "sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
        "criterion_id": "pool-target-structural-eligibility/v1",
        "passed": True,
        "raw_rows": 0, "prompt_ineligible": 0,
        "structurally_excluded": 0, "retained_rows": 0,
    }
    receipt.update(overrides)
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(receipt))
    return path


def test_the_screen_input_is_the_retained_population_not_the_raw_file(tmp_path: Path) -> None:
    """Eligibility precedes screening, so an ineligible prompt and a structurally
    excluded target must never reach the screen or its denominator."""
    rows = [
        _row("clean", ["pkg.a", "pkg.b"], "hello there", '[{"name": "pkg.a", "arguments": {}}]'),
        {"source_id": "no_tools", "messages": [
            {"role": "system", "content": "no tools at all"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": '[{"name": "x", "arguments": {}}]'}]},
        _row("bad_target", ["pkg.a"], "hi", "<tool_call>\n{'arguments': {}}\n</tool_call>"),
    ]
    pool = _pool(tmp_path, rows)
    art = build_artifact(pool, MANIFEST, _preflight(tmp_path, pool))

    assert art["prompt_ineligible"] == 1
    assert art["structurally_excluded"] == 1
    assert art["screen_input_rows"] == 1
    assert "3 cleaned source - 1 prompt-ineligible - 1 target-structural" in art["reconciliation"]


def test_weights_are_an_exact_integer_triple(tmp_path: Path) -> None:
    """No decimal is stored. P_std must come from exact ratios."""
    rows = [
        _row("m", ["zz.one", "zz.two"], "alpha beta", '[{"name": "zz.one", "arguments": {}}]'),
        _row("s", ["zz.solo"], "gamma delta", '[{"name": "zz.solo", "arguments": {}}]'),
    ]
    pool = _pool(tmp_path, rows)
    art = build_artifact(pool, MANIFEST, _preflight(tmp_path, pool))

    weights = art["weights"]
    assert set(weights) == {"n_multi", "n_single", "N"}
    assert all(isinstance(v, int) for v in weights.values())
    assert weights["N"] == weights["n_multi"] + weights["n_single"]
    assert not any(isinstance(v, float) for v in weights.values())


def test_the_artifact_binds_the_screened_question_files(tmp_path: Path) -> None:
    """A future BFCL re-pin must detectably invalidate this artifact rather than
    silently coexisting with it."""
    rows = [_row("a", ["qq.x"], "some text", '[{"name": "qq.x", "arguments": {}}]')]
    pool = _pool(tmp_path, rows)
    art = build_artifact(pool, MANIFEST, _preflight(tmp_path, pool))

    screened = art["screened_question_files"]
    assert len(screened) == 4
    for entry in screened:
        assert entry["role"] == "questions"
        assert len(entry["sha256"]) == 64
        assert {"category", "local_path", "sha256"} <= set(entry)
    assert len(art["manifest"]["sha256"]) == 64
    assert art["criterion_id"] == CRITERION_ID


def test_a_retained_row_that_cannot_be_screened_is_a_hard_failure(tmp_path: Path) -> None:
    """Not a new exclusion bucket -- the whole point of failing closed."""
    rows = [_row("a", ["qq.x"], "text", '[{"name": "qq.x", "arguments": {}}]')]
    pool = _pool(tmp_path, rows)
    preflight = _preflight(tmp_path, pool)

    import mining.decontaminate_pool as module

    class Exploding:
        def is_contaminated(self, *_a, **_k):
            raise RuntimeError("index unavailable")

        def screened_manifest(self):
            return []

    original = module.Decontaminator
    module.Decontaminator = lambda *_a, **_k: Exploding()
    try:
        with pytest.raises(DecontaminationError, match="could not be screened"):
            build_artifact(pool, MANIFEST, preflight)
    finally:
        module.Decontaminator = original


def test_a_retained_name_defect_stops_the_build(tmp_path: Path) -> None:
    """Discarding the parsed call names let a row calling a tool its prompt never
    presented pass straight into the survivor set. A name defect is a defect."""
    rows = [_row("bad", ["offered.tool"], "hi",
                 '[{"name": "invented.tool", "arguments": {}}]')]
    pool = _pool(tmp_path, rows)
    preflight = _preflight(tmp_path, pool)

    with pytest.raises(DecontaminationError, match="not present"):
        build_artifact(pool, MANIFEST, preflight)


def test_a_preflight_receipt_about_another_pool_is_refused(tmp_path: Path) -> None:
    """Binding only the criterion id would let the artifact cite an eligibility
    rule whose receipt was computed against different bytes."""
    rows = [_row("a", ["qq.x"], "text", '[{"name": "qq.x", "arguments": {}}]')]
    pool = _pool(tmp_path, rows)
    preflight = _preflight(tmp_path, pool, sha256="0" * 64)

    with pytest.raises(DecontaminationError, match="not this pool"):
        build_artifact(pool, MANIFEST, preflight)


def test_a_failed_preflight_refuses_to_screen(tmp_path: Path) -> None:
    rows = [_row("a", ["qq.x"], "text", '[{"name": "qq.x", "arguments": {}}]')]
    pool = _pool(tmp_path, rows)
    preflight = _preflight(tmp_path, pool, passed=False)

    with pytest.raises(DecontaminationError, match="passed=false"):
        build_artifact(pool, MANIFEST, preflight)


def test_the_committed_artifact_regenerates_exactly() -> None:
    """Arithmetic self-consistency is not enough: a hand-edited but coherent
    receipt would pass. Regenerate from the pinned inputs and compare."""
    path = REPO_ROOT / "mining" / "receipts" / "sft_dedup_v2_decontamination.json"
    committed = json.loads(path.read_text())
    regenerated = build_artifact(
        REPO_ROOT / "data" / "processed" / "sft_dedup_v2.jsonl",
        MANIFEST,
        REPO_ROOT / "mining" / "receipts" / "sft_dedup_v2_target_preflight.json",
    )
    assert regenerated == committed


def test_the_committed_artifact_reconciles(tmp_path: Path) -> None:
    """The real receipt: every row accounted for, weights summing to survivors."""
    art = json.loads(
        (REPO_ROOT / "mining" / "receipts" / "sft_dedup_v2_decontamination.json").read_text()
    )
    assert art["criterion_id"] == CRITERION_ID
    assert (
        art["prompt_ineligible"] + art["structurally_excluded"] + art["screen_input_rows"] == 12143
    )
    assert art["dropped"]["total"] + art["survivors"]["total"] == art["screen_input_rows"]
    w = art["weights"]
    assert (w["n_multi"], w["n_single"], w["N"]) == (
        art["survivors"]["multi"], art["survivors"]["single"], art["survivors"]["total"],
    )
