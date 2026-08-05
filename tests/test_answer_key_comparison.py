"""Tests for the two-answer-key comparison.

The comparison exists to support three public claims: that the two keys differ
at exactly one row, that the disputed row is concordant across candidates, and
that every paired contrast is therefore invariant to the key choice. A report
that stated those instead of measuring them would be worthless as evidence, so
these tests attack each one — a second differing row, a discordant row, a
mutated input file, a candidate missing items, and a stored flag that no longer
matches the committed scorer must all fail the run rather than produce a
confident-looking artifact.

They also attack the inputs' provenance, which is the quieter failure: a
generations file missing an entire candidate leaves every surviving candidate
complete and self-consistent, and a tampered exposure file moves a published
figure while the report's recorded inputs look untouched. Both are covered
here, at the evidence pin and again at the expected candidate set.

The fixtures are synthetic and tiny; the last test checks the committed report
still regenerates from the real pinned inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.answer_key_comparison import (
    ComparisonIntegrityError,
    apply_canonical_criterion,
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


def _question_rows(also_present: str | None = None) -> list[dict]:
    """Each item presents exactly the name the *pinned* key expects.

    That mirrors simple_python_363: the tool list offers the module-qualified
    name, the data-fix key expects it, and the release key asks for a bare tail
    the item never offered.

    `also_present` adds a second tool to the disputed item, which is what makes
    a *valid* disagreement expressible: two keys can then differ there while
    both name something the item actually offered.
    """
    rows = []
    for item_id in ITEM_IDS:
        names = [QUALIFIED_NAME if item_id == DISPUTED else f"fn_{item_id}"]
        if also_present is not None and item_id == DISPUTED:
            names.append(also_present)
        rows.append({
            "id": item_id,
            "question": [[{"role": "user", "content": item_id}]],
            "function": [{"name": name} for name in names],
        })
    return rows


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


CANDIDATES = ("dpo-50", "sft")
EXPOSURE = ("simple_python", "multiple")

# The adjudication the manifest carries. Tests that care about it override a
# field; every other test just needs the report to have one to record.
CRITERION = {
    "rule": "canonical = pinned AND valid; the preflight decides",
    "adjudicated_by": "owner",
    "adjudicated_on": "2026-08-05",
    "adjudication_ref": "test",
    "selected": "pinned_key",
    "selected_source_revision": DATAFIX_REVISION,
}

# A second exposure category, so a tampered file that feeds only the published
# exposure table — and no score — is still caught.
EXTRA_QUESTIONS = [
    {"id": "multiple_0", "function": [{"name": "circle.get"}, {"name": "triangle.get"}]},
    {"id": "multiple_1", "function": [{"name": "plain"}]},
]
EXTRA_ANSWERS = [
    {"id": "multiple_0", "ground_truth": [{"circle.get": {"x": ["1"]}}]},
    {"id": "multiple_1", "ground_truth": [{"plain": {"x": ["1"]}}]},
]


@pytest.fixture
def world(tmp_path: Path):
    """A self-contained repo root: manifest, both keys, generations, evidence index."""

    def build(
        *,
        release_name: str = UNQUALIFIED_NAME,
        extra_difference: str | None = None,
        emitted: dict[str, str] | None = None,
        drop: tuple[str, str] | None = None,
        drop_candidate: str | None = None,
        criterion: dict | None = None,
        also_present: str | None = None,
    ) -> tuple[Path, Path]:
        emitted = emitted or dict.fromkeys(CANDIDATES, QUALIFIED_NAME)
        data = tmp_path / "eval" / "bfcl_data"
        (data / "possible_answer").mkdir(parents=True)
        (data / "release_commit" / "possible_answer").mkdir(parents=True)

        payloads = {
            "eval/bfcl_data/BFCL_v4_simple_python.json": (
                _jsonl(_question_rows(also_present)), DATAFIX_REVISION, "simple_python",
                "questions"),
            "eval/bfcl_data/possible_answer/BFCL_v4_simple_python.json": (
                _jsonl(_key_rows(QUALIFIED_NAME)), DATAFIX_REVISION, "simple_python",
                "answer_key"),
            RELEASE_KEY_PATH: (
                _jsonl(_key_rows(release_name, extra_difference)),
                RELEASE_REVISION,
                "simple_python",
                "answer_key_release_commit",
            ),
            "eval/bfcl_data/BFCL_v4_multiple.json": (
                _jsonl(EXTRA_QUESTIONS), DATAFIX_REVISION, "multiple", "questions"),
            "eval/bfcl_data/possible_answer/BFCL_v4_multiple.json": (
                _jsonl(EXTRA_ANSWERS), DATAFIX_REVISION, "multiple", "answer_key"),
        }
        files = []
        for local_path, (payload, revision, category, role) in payloads.items():
            (tmp_path / local_path).write_bytes(payload)
            files.append(_spec(local_path, payload, revision, category, role))

        manifest_path = tmp_path / "eval" / "manifests" / "bfcl_v4_study2.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "upstream_repository": "ShishirPatil/gorilla",
                    "default_source_revision": DATAFIX_REVISION,
                    "canonical_key_criterion": dict(CRITERION, **(criterion or {})),
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
        if drop_candidate is not None:
            generations = [r for r in generations if r["model_name"] != drop_candidate]
        generations_path = tmp_path / "eval" / "results" / "generations.jsonl"
        generations_path.parent.mkdir(parents=True)
        payload = _jsonl(generations)
        generations_path.write_bytes(payload)

        # The evidence index pins what the run produced. It is written from the
        # generations as built, so a test that mutates the file afterwards is
        # attacking the pin rather than moving it too.
        (tmp_path / "eval" / "results" / "evidence.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_repository": "example/evidence",
                    "source_revision": "a" * 40,
                    "artifacts": [
                        {
                            "kind": "per-item generations",
                            "local_copy": "eval/results/generations.jsonl",
                            "sha256": _digest("sha256", payload),
                            "row_count": len(generations),
                        }
                    ],
                },
                indent=2,
            )
        )
        return manifest_path, generations_path

    build.root = tmp_path  # type: ignore[attr-defined]
    return build


def _build(world, **kwargs) -> dict:
    manifest_path, generations_path = world(**kwargs)
    return build_report(
        manifest_path,
        generations_path,
        world.root,
        evidence_path=world.root / "eval" / "results" / "evidence.json",
        expected_candidates=CANDIDATES,
        exposure_categories=EXPOSURE,
    )


def _report(world, **kwargs) -> dict:
    return _build(world, **kwargs)


def _rebuild(world, manifest_path: Path, generations_path: Path) -> dict:
    """Re-run against an already-built world, after a test has tampered with it."""
    return build_report(
        manifest_path,
        generations_path,
        world.root,
        evidence_path=world.root / "eval" / "results" / "evidence.json",
        expected_candidates=CANDIDATES,
        exposure_categories=EXPOSURE,
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
        _rebuild(world, manifest_path, generations_path)


def test_a_missing_pinned_input_fails_closed(world) -> None:
    manifest_path, generations_path = world()
    (world.root / RELEASE_KEY_PATH).unlink()

    with pytest.raises(ComparisonIntegrityError, match="missing pinned input"):
        _rebuild(world, manifest_path, generations_path)


def test_a_candidate_missing_an_item_fails_closed(world) -> None:
    """A short candidate would otherwise score against a smaller denominator."""
    manifest_path, generations_path = world(drop=("dpo-50", "item_1"))

    with pytest.raises(ComparisonIntegrityError, match="covers 2 of 3 items"):
        _rebuild(world, manifest_path, generations_path)


def _repin(world, generations_path: Path) -> None:
    """Re-point the evidence index at the current generations bytes.

    Used only by tests that need to get *past* the pin to attack a later check.
    """
    evidence_path = world.root / "eval" / "results" / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    payload = generations_path.read_bytes()
    artifact = evidence["artifacts"][0]
    artifact["sha256"] = _digest("sha256", payload)
    artifact["row_count"] = len(
        [line for line in payload.decode().splitlines() if line.strip()]
    )
    evidence_path.write_text(json.dumps(evidence, indent=2))


def test_generations_that_disagree_with_the_scorer_fail_closed(world) -> None:
    """The release column is only credible if the same code reproduces the
    pinned column the run actually recorded.

    The evidence pin is moved to match, so this attacks the drift check rather
    than stopping at the hash: a file can be perfectly pinned and still
    disagree with the scorer it claims to have been produced by.
    """
    manifest_path, generations_path = world()
    rows = [json.loads(line) for line in generations_path.read_text().splitlines() if line.strip()]
    rows[0]["overall_ok"] = not rows[0]["overall_ok"]
    generations_path.write_bytes(_jsonl(rows))
    _repin(world, generations_path)

    with pytest.raises(ComparisonIntegrityError, match="no longer agree"):
        _rebuild(world, manifest_path, generations_path)


# --------------------------------------------- provenance of the inputs ------


def test_a_whole_missing_candidate_fails_the_evidence_pin(world) -> None:
    """The dangerous case: every surviving candidate is complete and
    self-consistent, so nothing downstream notices an arm was dropped. The
    committed pin on the run's bytes is what notices."""
    manifest_path, generations_path = world()
    rows = [json.loads(line) for line in generations_path.read_text().splitlines() if line.strip()]
    generations_path.write_bytes(_jsonl([r for r in rows if r["model_name"] != "dpo-50"]))

    with pytest.raises(ComparisonIntegrityError, match="sha256"):
        _rebuild(world, manifest_path, generations_path)


def test_a_whole_missing_candidate_fails_even_when_the_pin_agrees(world) -> None:
    """Second line of defence: a short run pinned to its own short output is
    still missing an arm, and the expected candidate set says so."""
    manifest_path, generations_path = world(drop_candidate="dpo-50")

    with pytest.raises(ComparisonIntegrityError, match="expects \\['dpo-50', 'sft'\\]"):
        _rebuild(world, manifest_path, generations_path)


def test_an_unexpected_extra_candidate_fails_closed(world) -> None:
    manifest_path, generations_path = world(
        emitted={"sft": QUALIFIED_NAME, "dpo-50": QUALIFIED_NAME, "dpo-999": QUALIFIED_NAME}
    )
    _repin(world, generations_path)

    with pytest.raises(ComparisonIntegrityError, match="unexpected=\\['dpo-999'\\]"):
        _rebuild(world, manifest_path, generations_path)


def test_generations_not_named_in_the_evidence_index_fail_closed(world) -> None:
    """Fingerprinting an arbitrary file records a hash; it does not establish
    the file is the study-1 run's output."""
    manifest_path, generations_path = world()
    stranger = generations_path.with_name("some_other_generations.jsonl")
    stranger.write_bytes(generations_path.read_bytes())

    with pytest.raises(ComparisonIntegrityError, match="pinned exactly once"):
        _rebuild(world, manifest_path, stranger)


def test_a_tampered_exposure_input_fails_closed(world) -> None:
    """The exposure figures feed the prereg amendment and the upstream issue,
    so they must be as traceable as the scores — even though no score reads
    the `multiple` files."""
    manifest_path, generations_path = world()
    path = world.root / "eval/bfcl_data/BFCL_v4_multiple.json"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows[0]["function"][0]["name"] = "tampered.name"
    path.write_bytes(_jsonl(rows))

    with pytest.raises(ComparisonIntegrityError, match="multiple/questions"):
        _rebuild(world, manifest_path, generations_path)


def test_every_exposure_input_is_recorded_in_the_report(world) -> None:
    report = _report(world)
    exposure = report["inputs"]["exposure"]

    assert set(exposure) == set(EXPOSURE)
    for category in EXPOSURE:
        for role in ("questions", "answer_key"):
            entry = exposure[category][role]
            assert entry["sha256"] and entry["path"] and entry["source_revision"]
    assert set(report["qualified_name_exposure"]) == set(EXPOSURE)


def test_the_evidence_index_provenance_is_recorded(world) -> None:
    report = _report(world)
    index = report["inputs"]["evidence_index"]

    assert index["pinned_row_count"] == 6
    assert index["source_repository"] == "example/evidence"
    assert report["inputs"]["generations"]["verified_against"] == index["path"]


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


# ------------------------------------------------ the canonical adjudication ----
#
# The owner picked the key that scores higher, one message after instructing us
# to take the lower number. The defence is that the criterion is stated without
# reference to the score and is applied by machine -- so these tests are mostly
# about the rule still selecting correctly when the measurements are inverted.


def test_the_criterion_selects_the_only_valid_key() -> None:
    selection = apply_canonical_criterion(
        {
            "pinned_key": {"consistent": True, "items_checked": 400, "defect": None},
            "release_key": {"consistent": False, "items_checked": 400, "defect": "..."},
        }
    )
    assert selection["decided"] and selection["selected"] == "pinned_key"


def test_the_criterion_is_outcome_independent() -> None:
    """Invert which key is valid and the same rule hands back the other one.
    If this ever needed a special case for the key that scores better, the rule
    would be a rationalisation."""
    selection = apply_canonical_criterion(
        {
            "pinned_key": {"consistent": False, "items_checked": 400, "defect": "..."},
            "release_key": {"consistent": True, "items_checked": 400, "defect": None},
        }
    )
    assert selection["decided"] and selection["selected"] == "release_key"


@pytest.mark.parametrize("both", [True, False])
def test_the_criterion_refuses_to_decide_when_it_cannot_discriminate(both: bool) -> None:
    selection = apply_canonical_criterion(
        {
            "pinned_key": {"consistent": both, "items_checked": 400, "defect": None},
            "release_key": {"consistent": both, "items_checked": 400, "defect": None},
        }
    )
    assert not selection["decided"] and selection["selected"] is None


def test_the_report_records_the_adjudication_with_both_key_hashes(world) -> None:
    report = _report(world)
    check_report_invariants(report)

    adjudication = report["adjudication"]
    assert adjudication["selected"] == "pinned_key"
    assert adjudication["derived_from_measurement"]["selected"] == "pinned_key"
    assert adjudication["candidate_keys"]["pinned_key"]["source_revision"] == DATAFIX_REVISION
    assert adjudication["candidate_keys"]["release_key"]["source_revision"] == RELEASE_REVISION
    assert adjudication["candidate_keys"]["pinned_key"]["sha256"] != (
        adjudication["candidate_keys"]["release_key"]["sha256"]
    )


def test_a_recorded_choice_the_rule_does_not_produce_fails_closed(world) -> None:
    """The artifact is the permanent record of *why* this key is canonical. A
    manifest that names the other key while quoting this rule is the failure
    mode worth catching: the prose would still read as principled."""
    report = _report(world, criterion={"selected": "release_key",
                                       "selected_source_revision": RELEASE_REVISION})
    with pytest.raises(ComparisonIntegrityError, match="the rule and the choice have come apart"):
        check_report_invariants(report)


def test_an_adjudication_naming_the_wrong_revision_fails_closed(world) -> None:
    report = _report(world, criterion={"selected_source_revision": RELEASE_REVISION})
    with pytest.raises(ComparisonIntegrityError, match="but pinned_key is pinned at"):
        check_report_invariants(report)


def test_a_manifest_with_no_criterion_cannot_produce_the_artifact(world) -> None:
    manifest_path, generations_path = world()
    manifest = json.loads(manifest_path.read_text())
    del manifest["canonical_key_criterion"]
    manifest_path.write_text(json.dumps(manifest, indent=2))
    with pytest.raises(ComparisonIntegrityError, match="declares no `canonical_key_criterion`"):
        _rebuild(world, manifest_path, generations_path)


def test_the_criterion_cannot_decide_when_both_keys_are_valid(world) -> None:
    """Two keys that disagree while both naming a tool the item actually offers.
    The preflight has nothing to say, so the rule must refuse rather than let
    the standing choice coast through on a check that did not discriminate."""
    report = _report(world, release_name="second.fn", also_present="second.fn")
    assert report["key_difference"]["differing_row_count"] == 1
    assert all(k["consistent"] for k in report["key_internal_consistency"].values())
    with pytest.raises(ComparisonIntegrityError, match="does not decide"):
        check_report_invariants(report)
