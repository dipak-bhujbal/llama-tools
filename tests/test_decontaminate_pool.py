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
from mining.pool_strata import target_defects

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
    """A matching eligibility receipt, carrying this pool's *real* counts.

    An earlier version wrote zeros and was accepted, which is exactly the gap it
    should have been proving does not exist: the producer re-derives these and
    now refuses a receipt that describes a different population.
    """
    report = target_defects(pool)
    receipt = {
        "sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
        "criterion_id": "pool-target-structural-eligibility/v1",
        "passed": True,
        "raw_rows": report["raw_rows"],
        "prompt_ineligible": report["prompt_ineligible"],
        "structurally_excluded": report["structurally_excluded"],
        "retained_rows": report["retained_rows"],
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
    # 5 since Decision C added live_multiple to the manifest: the pool is
    # screened against every set that will ever be scored *or selected on*.
    assert len(screened) == 5
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

    with pytest.raises(DecontaminationError, match="which is not True"):
        build_artifact(pool, MANIFEST, preflight)


def test_the_current_artifact_regenerates_exactly() -> None:
    """Arithmetic self-consistency is not enough: a hand-edited but coherent
    receipt would pass. Regenerate from the pinned inputs and compare.

    The *current* artifact is the live_multiple one: Decision C added a fifth
    screened category, so the pool the miner consumes is the re-screened pool.
    """
    path = (
        REPO_ROOT / "mining" / "receipts"
        / "sft_dedup_v2_decontamination_with_live_multiple.json"
    )
    committed = json.loads(path.read_text())
    regenerated = build_artifact(
        REPO_ROOT / "data" / "processed" / "sft_dedup_v2.jsonl",
        MANIFEST,
        REPO_ROOT / "mining" / "receipts" / "sft_dedup_v2_target_preflight.json",
    )
    assert regenerated == committed


def test_the_superseded_artifact_is_kept_and_is_visibly_superseded() -> None:
    """The pre-Decision-C artifact stays on disk, byte-for-byte, forever.

    It is evidence of what was screened when §2 was frozen, and it is *not*
    regenerable from today's manifest -- which is the honest state of affairs
    and must be detectable rather than inferred. If its recorded manifest digest
    ever equals the current one, either it was edited or the supersession was
    undone; both are defects.
    """
    superseded = json.loads(
        (REPO_ROOT / "mining" / "receipts" / "sft_dedup_v2_decontamination.json").read_text()
    )
    current = json.loads(
        (
            REPO_ROOT / "mining" / "receipts"
            / "sft_dedup_v2_decontamination_with_live_multiple.json"
        ).read_text()
    )

    assert superseded["criterion_id"] == current["criterion_id"] == CRITERION_ID
    assert superseded["manifest"]["sha256"] != current["manifest"]["sha256"]
    assert len(superseded["screened_question_files"]) == 4
    assert len(current["screened_question_files"]) == 5
    # The weights the amendment moves, stated as data rather than as prose.
    def triple(artifact: dict) -> tuple[int, int, int]:
        weights = artifact["weights"]
        return weights["n_multi"], weights["n_single"], weights["N"]

    assert triple(superseded) == (8173, 2997, 11170)
    assert triple(current) == (8081, 2990, 11071)


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


def test_a_receipt_whose_counts_disagree_is_refused(tmp_path: Path) -> None:
    """Embedding the counts without comparing them let the artifact cite a
    receipt describing a different population."""
    rows = [_row("a", ["qq.x"], "text", '[{"name": "qq.x", "arguments": {}}]')]
    pool = _pool(tmp_path, rows)
    preflight = _preflight(tmp_path, pool, retained_rows=99)

    with pytest.raises(DecontaminationError, match="disagrees with the re-derived population"):
        build_artifact(pool, MANIFEST, preflight)


@pytest.mark.parametrize("value", ["false", "true", 1, "yes"])
def test_a_non_boolean_passed_cannot_open_the_gate(tmp_path: Path, value) -> None:
    """A fail-closed gate must not be opened by a truthy string or integer."""
    rows = [_row("a", ["qq.x"], "text", '[{"name": "qq.x", "arguments": {}}]')]
    pool = _pool(tmp_path, rows)
    preflight = _preflight(tmp_path, pool, passed=value)

    with pytest.raises(DecontaminationError, match="which is not True"):
        build_artifact(pool, MANIFEST, preflight)
