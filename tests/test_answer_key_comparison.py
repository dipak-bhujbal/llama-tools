"""Tests for the two-answer-key comparison.

The comparison exists to support three public claims: that the two keys differ
at exactly one row, that the disputed row is concordant across candidates, and
that every paired contrast is therefore invariant to the key choice. A report
that stated those instead of measuring them would be worthless as evidence, so
these tests attack each one — a second differing row, a discordant row, a
mutated input file, a candidate missing items, and a stored flag that no longer
matches the committed scorer must all fail the run rather than produce a
confident-looking artifact.

The fixtures are synthetic and tiny; the last test checks the committed report
still regenerates from the real pinned inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.answer_key_comparison import (
    ComparisonIntegrityError,
    build_report,
    check_report_invariants,
)
from eval.bfcl_scoring import score
from eval.fetch_pinned_bfcl import _digest, _git_blob_sha1

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_KEY_PATH = "eval/bfcl_data/release_commit/possible_answer/BFCL_v4_simple_python.json"
RELEASE_REVISION = "58f57e9124ea981403792dd51e00a6577e621fae"
DATAFIX_REVISION = "9d8416a96d1d69975493f1b6d60ff07d12a1726a"

# Three items; the third is the "disputed" one whose key name differs between
# revisions, mirroring simple_python_363.
ITEM_IDS = ("item_0", "item_1", "item_2")
DISPUTED = "item_2"
QUALIFIED_NAME = "restaurant_search.find_closest"
UNQUALIFIED_NAME = "find_closest"


def _jsonl(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()


def _spec(local_path: str, payload: bytes, revision: str, category: str, role: str) -> dict:
    ids = [json.loads(line)["id"] for line in payload.decode().splitlines() if line.strip()]
    sorted_ids = ("\n".join(sorted(ids)) + "\n").encode()
    return {
        "category": category,
        "role": role,
        "source_revision": revision,
        "upstream_path": f"upstream/{Path(local_path).name}",
        "local_path": local_path,
        "git_blob_sha1": _git_blob_sha1(payload),
        "sha256": _digest("sha256", payload),
        "row_count": len(ids),
        "unique_id_count": len(set(ids)),
        "sorted_id_sha256": _digest("sha256", sorted_ids),
    }


def _question_rows() -> list[dict]:
    """Each item presents exactly the name the *pinned* key expects.

    That mirrors simple_python_363: the tool list offers the module-qualified
    name, the data-fix key expects it, and the release key asks for a bare tail
    the item never offered.
    """
    return [
        {
            "id": item_id,
            "question": [[{"role": "user", "content": item_id}]],
            "function": [
                {"name": QUALIFIED_NAME if item_id == DISPUTED else f"fn_{item_id}"}
            ],
        }
        for item_id in ITEM_IDS
    ]


def _key_rows(name_at_disputed: str, extra_difference: str | None = None) -> list[dict]:
    rows = []
    for item_id in ITEM_IDS:
        name = name_at_disputed if item_id == DISPUTED else f"fn_{item_id}"
        if extra_difference is not None and item_id == "item_1":
            name = extra_difference
        rows.append({"id": item_id, "ground_truth": [{name: {"city": ["Boston"]}}]})
    return rows


def _generation_rows(emitted: dict[str, dict[str, str]], key_rows: list[dict]) -> list[dict]:
    """Build generations whose stored flags come from the real scorer.

    `emitted` is {candidate: {item_id: function_name}}. Storing anything the
    scorer would not produce is the exact drift the comparison refuses to
    tolerate, so the fixture derives the flags rather than hard-coding them.
    """
    key = {row["id"]: row["ground_truth"][0] for row in key_rows}
    rows = []
    for candidate, per_item in emitted.items():
        for item_id, name in per_item.items():
            parsed = {"name": name, "arguments": {"city": "Boston"}}
            name_ok, args_ok, overall_ok, reason = score(parsed, key[item_id])
            rows.append({
                "id": item_id,
                "model_name": candidate,
                "output": json.dumps(parsed),
                "parsed_name": name,
                "parsed_args": parsed["arguments"],
                "name_ok": name_ok,
                "args_ok": args_ok,
                "overall_ok": overall_ok,
                "failure_reason": reason,
                "json_valid": True,
            })
    return rows


def _emitted_all(name_at_disputed: dict[str, str]) -> dict[str, dict[str, str]]:
    """Every candidate answers every item correctly except as directed at the
    disputed item, where each candidate emits the name it is given."""
    return {
        candidate: {
            item_id: (name if item_id == DISPUTED else f"fn_{item_id}") for item_id in ITEM_IDS
        }
        for candidate, name in name_at_disputed.items()
    }


@pytest.fixture
def world(tmp_path: Path):
    """A self-contained repo root with a manifest, both keys, and generations."""

    def build(
        *,
        release_name: str = UNQUALIFIED_NAME,
        extra_difference: str | None = None,
        emitted: dict[str, str] | None = None,
        drop: tuple[str, str] | None = None,
    ) -> tuple[Path, Path]:
        emitted = emitted or {"sft": QUALIFIED_NAME, "dpo-50": QUALIFIED_NAME}
        data = tmp_path / "eval" / "bfcl_data"
        (data / "possible_answer").mkdir(parents=True)
        (data / "release_commit" / "possible_answer").mkdir(parents=True)

        payloads = {
            "eval/bfcl_data/BFCL_v4_simple_python.json": (
                _jsonl(_question_rows()), DATAFIX_REVISION, "questions"),
            "eval/bfcl_data/possible_answer/BFCL_v4_simple_python.json": (
                _jsonl(_key_rows(QUALIFIED_NAME)), DATAFIX_REVISION, "answer_key"),
            RELEASE_KEY_PATH: (
                _jsonl(_key_rows(release_name, extra_difference)),
                RELEASE_REVISION,
                "answer_key_release_commit",
            ),
        }
        files = []
        for local_path, (payload, revision, role) in payloads.items():
            (tmp_path / local_path).write_bytes(payload)
            files.append(_spec(local_path, payload, revision, "simple_python", role))

        manifest_path = tmp_path / "eval" / "manifests" / "bfcl_v4_study2.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "upstream_repository": "ShishirPatil/gorilla",
                    "default_source_revision": DATAFIX_REVISION,
                    "files": files,
                },
                indent=2,
            )
        )

        generations = _generation_rows(_emitted_all(emitted), _key_rows(QUALIFIED_NAME))
        if drop is not None:
            candidate, item_id = drop
            generations = [
                row
                for row in generations
                if not (row["model_name"] == candidate and row["id"] == item_id)
            ]
        generations_path = tmp_path / "eval" / "results" / "generations.jsonl"
        generations_path.parent.mkdir(parents=True)
        generations_path.write_bytes(_jsonl(generations))
        return manifest_path, generations_path

    build.root = tmp_path  # type: ignore[attr-defined]
    return build


def _report(world, **kwargs) -> dict:
    manifest_path, generations_path = world(**kwargs)
    return build_report(
        manifest_path, generations_path, world.root, include_exposure=False
    )


# ------------------------------------------------------------- happy path ----


def test_one_differing_row_is_measured_not_assumed(world) -> None:
    report = _report(world)

    assert report["key_difference"]["differing_row_count"] == 1
    assert report["key_difference"]["differing_ids"] == [DISPUTED]
    detail = report["key_difference"]["detail"][DISPUTED]
    assert detail["pinned_key_expects"] == [QUALIFIED_NAME]
    assert detail["release_key_expects"] == [UNQUALIFIED_NAME]
    check_report_invariants(report)


def test_release_key_score_is_rescored_not_patched(world) -> None:
    """Both candidates emit the qualified name, so each loses exactly the
    disputed item under the release key — derived by running the scorer."""
    report = _report(world)

    assert report["scores"]["pinned_key"] == {"sft": 3, "dpo-50": 3}
    assert report["scores"]["release_key"] == {"sft": 2, "dpo-50": 2}
    assert report["recomputation_check"] == {"rows_rechecked": 6, "disagreements": 0}


def test_contrasts_are_identical_under_both_keys(world) -> None:
    report = _report(world)
    contrasts = report["contrasts"]

    assert contrasts["invariance"]["identical"] is True
    assert contrasts["pinned_key"] == contrasts["release_key"]
    assert contrasts["pinned_key"]["sft_vs_dpo-50"]["discordant"] == 0


# ------------------------------------------------------- attacked claims ----


def test_a_second_differing_row_fails_the_run(world) -> None:
    """The one-row scope is what makes the concordance argument sufficient."""
    report = _report(world, extra_difference="renamed_fn")

    assert report["key_difference"]["differing_row_count"] == 2
    with pytest.raises(ComparisonIntegrityError, match="differ at 2 rows"):
        check_report_invariants(report)


def test_a_discordant_disputed_item_fails_the_run(world) -> None:
    """If candidates disagree at the disputed row, the key choice does move the
    paired tests, and the invariance claim must not survive."""
    report = _report(world, emitted={"sft": QUALIFIED_NAME, "dpo-50": UNQUALIFIED_NAME})

    assert report["disputed_items"][DISPUTED]["concordant_under_pinned_key"] is False
    with pytest.raises(ComparisonIntegrityError, match="not concordant across candidates"):
        check_report_invariants(report)


def test_discordant_sets_that_move_are_reported_as_a_difference(world) -> None:
    report = _report(world, emitted={"sft": QUALIFIED_NAME, "dpo-50": UNQUALIFIED_NAME})
    invariance = report["contrasts"]["invariance"]

    assert invariance["identical"] is False
    assert "sft_vs_dpo-50" in invariance["differences"]


def test_changed_input_bytes_fail_closed(world) -> None:
    manifest_path, generations_path = world()
    key_path = world.root / "eval/bfcl_data/possible_answer/BFCL_v4_simple_python.json"
    rows = _key_rows(QUALIFIED_NAME)
    rows[0]["ground_truth"] = [{"tampered": {"city": ["Boston"]}}]
    key_path.write_bytes(_jsonl(rows))

    with pytest.raises(ComparisonIntegrityError, match="sha256"):
        build_report(manifest_path, generations_path, world.root, include_exposure=False)


def test_a_missing_pinned_input_fails_closed(world) -> None:
    manifest_path, generations_path = world()
    (world.root / RELEASE_KEY_PATH).unlink()

    with pytest.raises(ComparisonIntegrityError, match="missing pinned input"):
        build_report(manifest_path, generations_path, world.root, include_exposure=False)


def test_a_candidate_missing_an_item_fails_closed(world) -> None:
    """A short candidate would otherwise score against a smaller denominator."""
    manifest_path, generations_path = world(drop=("dpo-50", "item_1"))

    with pytest.raises(ComparisonIntegrityError, match="covers 2 of 3 items"):
        build_report(manifest_path, generations_path, world.root, include_exposure=False)


def test_generations_that_disagree_with_the_scorer_fail_closed(world) -> None:
    """The release column is only credible if the same code reproduces the
    pinned column the run actually recorded."""
    manifest_path, generations_path = world()
    rows = [json.loads(line) for line in generations_path.read_text().splitlines() if line.strip()]
    rows[0]["overall_ok"] = not rows[0]["overall_ok"]
    generations_path.write_bytes(_jsonl(rows))

    with pytest.raises(ComparisonIntegrityError, match="no longer agree"):
        build_report(manifest_path, generations_path, world.root, include_exposure=False)


# ------------------------------------------------------- committed report ----


@pytest.mark.skipif(
    not (REPO_ROOT / RELEASE_KEY_PATH).is_file(),
    reason="pinned BFCL data is gitignored; run `python eval/fetch_pinned_bfcl.py` first",
)
def test_committed_report_regenerates_byte_for_byte() -> None:
    """The artifact in eval/results is reproducible from the pinned inputs."""
    committed = json.loads((REPO_ROOT / "eval/results/answer_key_comparison.json").read_text())
    report = build_report()
    check_report_invariants(report)

    assert report == committed


def test_internal_consistency_is_measured_for_each_key(world) -> None:
    """The pinned key only ever expects names the item presented; the release
    key does not, which is the fact that decides which key is defensible."""
    report = _report(world)
    consistency = report["key_internal_consistency"]

    assert consistency["pinned_key"] == {"consistent": True, "items_checked": 3, "defect": None}
    assert consistency["release_key"]["consistent"] is False
    assert DISPUTED in consistency["release_key"]["defect"]
    assert UNQUALIFIED_NAME in consistency["release_key"]["defect"]
