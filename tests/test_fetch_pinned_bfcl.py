from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval.fetch_pinned_bfcl import (
    VerificationError,
    _git_blob_sha1,
    load_manifest,
    preflight_manifest_keys,
    verify_payload,
)


def _payload() -> bytes:
    return b'{"id": "b", "value": 2}\n{"id": "a", "value": 1}\n'


def _spec(payload: bytes) -> dict:
    sorted_ids = b"a\nb\n"
    return {
        "local_path": "eval/bfcl_data/example.json",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "git_blob_sha1": _git_blob_sha1(payload),
        "row_count": 2,
        "unique_id_count": 2,
        "sorted_id_sha256": hashlib.sha256(sorted_ids).hexdigest(),
    }


def test_verify_payload_accepts_all_frozen_invariants() -> None:
    payload = _payload()
    verify_payload(payload, _spec(payload))


def test_verify_payload_rejects_changed_bytes() -> None:
    payload = _payload()
    with pytest.raises(VerificationError, match="sha256"):
        verify_payload(payload + b"\n", _spec(payload))


def test_verify_payload_rejects_duplicate_ids_even_with_valid_byte_hashes() -> None:
    payload = b'{"id": "a"}\n{"id": "a"}\n'
    spec = _spec(payload)
    spec["unique_id_count"] = 2
    spec["sorted_id_sha256"] = hashlib.sha256(b"a\na\n").hexdigest()
    with pytest.raises(VerificationError, match="unique ids"):
        verify_payload(payload, spec)


def test_committed_manifest_is_valid_json() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "eval"
        / "manifests"
        / "bfcl_v4_study2.json"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["default_source_revision"] == (
        "9d8416a96d1d69975493f1b6d60ff07d12a1726a"
    )
    assert len(manifest["files"]) == 8
    assert all("source_revision" in spec for spec in manifest["files"])
    # Every category carrying pinned files is classified, so the standing
    # preflight can never pass a category over for lack of an opinion.
    assert set(manifest["categories"]) == {
        spec["category"] for spec in manifest["files"]
    }


def test_the_release_commit_answer_key_is_pinned_at_the_release_revision() -> None:
    """The disputed-key comparison needs the release key as a verified input,
    not a stated value, so it is pinned like every other frozen file — and at
    the release commit, not the data-fix revision the rest of the set uses."""
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "eval"
        / "manifests"
        / "bfcl_v4_study2.json"
    )
    manifest = json.loads(manifest_path.read_text())
    specs = [
        spec for spec in manifest["files"] if spec["role"] == "answer_key_release_commit"
    ]

    assert len(specs) == 1
    spec = specs[0]
    assert spec["source_revision"] == manifest["upstream_release_commit"]
    assert spec["source_revision"] != manifest["default_source_revision"]
    assert spec["sha256"] == (
        "38b1bc7469d1de73a812ffce9e2b10a1d8812425fd090ed314066ccec76d0ceb"
    )
    assert spec["row_count"] == 400

    canonical = [
        s
        for s in manifest["files"]
        if s["category"] == "simple_python" and s["role"] == "answer_key"
    ]
    assert len(canonical) == 1, "the canonical key must stay unambiguous"
    assert canonical[0]["sha256"] != spec["sha256"]


# --------------------------------------------- the standing answer-name preflight ----
#
# `simple_python_363` was found by hand. These cover the machinery that now
# looks for its cousins on every manifest verification, and in particular the
# ways a category could go unchecked while the run still reported success.


def _q(item_id: str, *names: str) -> dict:
    return {"id": item_id, "function": [{"name": name} for name in names]}


def _a(item_id: str, name: str) -> dict:
    return {"id": item_id, "ground_truth": [{name: {}}]}


def _payloads(**by_category: tuple[list[dict], list[dict] | None]) -> dict:
    out: dict[tuple[str, str], bytes] = {}
    for category, (questions, answers) in by_category.items():
        out[(category, "questions")] = _jsonl(questions)
        if answers is not None:
            out[(category, "answer_key")] = _jsonl(answers)
    return out


def _jsonl(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(row) for row in rows) + "\n").encode()


def _manifest(categories: dict, files: list[dict]) -> dict:
    return {"schema_version": 2, "categories": categories, "files": files}


def _file(category: str, role: str) -> dict:
    return {"category": category, "role": role, "local_path": f"{category}.{role}.json"}


def test_preflight_runs_over_every_pinned_category() -> None:
    manifest = _manifest(
        {"alpha": {"answer_key_policy": "required"}, "beta": {"answer_key_policy": "required"}},
        [_file(c, r) for c in ("alpha", "beta") for r in ("questions", "answer_key")],
    )
    payloads = _payloads(
        alpha=([_q("a1", "pkg.fn")], [_a("a1", "pkg.fn")]),
        beta=([_q("b1", "other.fn")], [_a("b1", "other.fn")]),
    )
    receipts = preflight_manifest_keys(manifest, payloads)
    assert {r["category"]: r["status"] for r in receipts} == {"alpha": "clean", "beta": "clean"}
    assert sum(r["items_checked"] for r in receipts) == 2


def test_a_key_defect_in_any_category_fails_the_manifest() -> None:
    """The defect that produced simple_python_363, in a category that is not
    simple_python. Verifying bytes would pass this; the whole point is that it
    does not."""
    manifest = _manifest(
        {"multiple": {"answer_key_policy": "required"}},
        [_file("multiple", "questions"), _file("multiple", "answer_key")],
    )
    payloads = _payloads(multiple=([_q("m1", "restaurant.find")], [_a("m1", "find")]))
    with pytest.raises(VerificationError, match="never presented"):
        preflight_manifest_keys(manifest, payloads)


def test_a_category_with_no_declared_policy_fails_rather_than_being_skipped() -> None:
    manifest = _manifest(
        {"alpha": {"answer_key_policy": "required"}},
        [_file("alpha", "questions"), _file("alpha", "answer_key"), _file("ghost", "questions")],
    )
    payloads = _payloads(
        alpha=([_q("a1", "pkg.fn")], [_a("a1", "pkg.fn")]), ghost=([_q("g1", "x")], None)
    )
    with pytest.raises(VerificationError, match="declares no answer-key policy"):
        preflight_manifest_keys(manifest, payloads)


def test_a_keyless_category_is_recorded_as_deliberate_not_absent() -> None:
    """`irrelevance` ships no answer key by schema. That must be a declaration
    in the manifest, not an inference from the file list -- otherwise a key that
    goes missing looks exactly like a key that never existed."""
    manifest = _manifest(
        {"irrelevance": {"answer_key_policy": "none_by_schema", "note": "no possible_answer"}},
        [_file("irrelevance", "questions")],
    )
    receipts = preflight_manifest_keys(manifest, _payloads(irrelevance=([_q("i1", "x")], None)))
    assert [r["status"] for r in receipts] == ["no-answer-key-by-schema"]


def test_a_required_key_that_goes_missing_fails_closed() -> None:
    manifest = _manifest(
        {"alpha": {"answer_key_policy": "required"}}, [_file("alpha", "questions")]
    )
    with pytest.raises(VerificationError, match="pins no answer_key"):
        preflight_manifest_keys(manifest, _payloads(alpha=([_q("a1", "pkg.fn")], None)))


def test_a_keyless_declaration_that_gains_a_key_fails_closed() -> None:
    manifest = _manifest(
        {"irrelevance": {"answer_key_policy": "none_by_schema"}},
        [_file("irrelevance", "questions"), _file("irrelevance", "answer_key")],
    )
    payloads = _payloads(irrelevance=([_q("i1", "x")], [_a("i1", "x")]))
    with pytest.raises(VerificationError, match="one of the two is wrong"):
        preflight_manifest_keys(manifest, payloads)


def test_an_unknown_policy_fails_closed() -> None:
    manifest = _manifest(
        {"alpha": {"answer_key_policy": "probably_fine"}},
        [_file("alpha", "questions"), _file("alpha", "answer_key")],
    )
    payloads = _payloads(alpha=([_q("a1", "pkg.fn")], [_a("a1", "pkg.fn")]))
    with pytest.raises(VerificationError, match="unknown answer_key_policy"):
        preflight_manifest_keys(manifest, payloads)


def test_a_manifest_with_no_categories_block_refuses_to_guess() -> None:
    manifest = {"schema_version": 2, "files": [_file("alpha", "questions")]}
    with pytest.raises(VerificationError, match="declares no `categories` block"):
        preflight_manifest_keys(manifest, _payloads(alpha=([_q("a1", "x")], None)))


def test_the_comparison_only_key_must_keep_failing_the_preflight() -> None:
    """The release key is retained *because* it is defective. If it ever passed,
    the adjudication that rejected it would no longer follow, so silence here
    would be the wrong answer."""
    spec = dict(_file("simple_python", "answer_key_release_commit"))
    spec["preflight_expectation"] = "defective"
    manifest = _manifest(
        {"simple_python": {"answer_key_policy": "required"}},
        [_file("simple_python", "questions"), _file("simple_python", "answer_key"), spec],
    )
    payloads = _payloads(simple_python=([_q("s1", "pkg.fn")], [_a("s1", "pkg.fn")]))
    payloads[("simple_python", "answer_key_release_commit")] = _jsonl([_a("s1", "pkg.fn")])
    with pytest.raises(VerificationError, match="now passes the answer-name preflight"):
        preflight_manifest_keys(manifest, payloads)


def test_a_comparison_only_key_with_no_stated_expectation_fails_closed() -> None:
    manifest = _manifest(
        {"simple_python": {"answer_key_policy": "required"}},
        [
            _file("simple_python", "questions"),
            _file("simple_python", "answer_key"),
            _file("simple_python", "answer_key_release_commit"),
        ],
    )
    payloads = _payloads(simple_python=([_q("s1", "pkg.fn")], [_a("s1", "pkg.fn")]))
    payloads[("simple_python", "answer_key_release_commit")] = _jsonl([_a("s1", "find")])
    with pytest.raises(VerificationError, match="preflight_expectation"):
        preflight_manifest_keys(manifest, payloads)


def test_a_schema_version_1_manifest_is_refused(tmp_path: Path) -> None:
    """v1 has no per-category policy, so loading one would mean skipping every
    category silently -- the exact failure this schema bump exists to remove."""
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"schema_version": 1, "files": [_file("alpha", "questions")]}))
    with pytest.raises(VerificationError, match="predates the standing answer-name preflight"):
        load_manifest(path)


# ------------------------------------------------- duplicate (category, role) ----
#
# Every consumer resolves files by (category, role) into a dict. Two specs
# sharing a pair means the first is pinned and the second silently wins, so a
# defective key can be pinned, verified, and never preflighted.


def test_duplicate_answer_keys_are_rejected_not_collapsed() -> None:
    manifest = _manifest(
        {"alpha": {"answer_key_policy": "required"}},
        [_file("alpha", "questions"), _file("alpha", "answer_key"), _file("alpha", "answer_key")],
    )
    payloads = _payloads(alpha=([_q("a1", "pkg.fn")], [_a("a1", "pkg.fn")]))
    with pytest.raises(VerificationError, match="pinned twice"):
        preflight_manifest_keys(manifest, payloads)


def test_a_defective_key_cannot_hide_behind_a_clean_duplicate() -> None:
    """The reason the duplicate check exists. Pin a defective key first and a
    clean one second: last-write-wins would preflight only the clean one and
    report a pass the defective key never earned."""
    defective = dict(_file("alpha", "answer_key"), local_path="defective.json")
    clean = dict(_file("alpha", "answer_key"), local_path="clean.json")
    manifest = _manifest(
        {"alpha": {"answer_key_policy": "required"}},
        [_file("alpha", "questions"), defective, clean],
    )
    payloads = _payloads(alpha=([_q("a1", "pkg.fn")], [_a("a1", "pkg.fn")]))
    with pytest.raises(VerificationError, match=r"defective\.json and clean\.json"):
        preflight_manifest_keys(manifest, payloads)


def test_duplicate_question_files_are_rejected() -> None:
    manifest = _manifest(
        {"alpha": {"answer_key_policy": "required"}},
        [_file("alpha", "questions"), _file("alpha", "questions"), _file("alpha", "answer_key")],
    )
    payloads = _payloads(alpha=([_q("a1", "pkg.fn")], [_a("a1", "pkg.fn")]))
    with pytest.raises(VerificationError, match="alpha/questions is pinned twice"):
        preflight_manifest_keys(manifest, payloads)


def test_load_manifest_rejects_duplicates_before_any_download(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            _manifest(
                {"alpha": {"answer_key_policy": "required"}},
                [_file("alpha", "questions"), _file("alpha", "questions")],
            )
        )
    )
    with pytest.raises(VerificationError, match="pinned twice"):
        load_manifest(path)


def test_the_committed_manifest_has_no_duplicate_roles() -> None:
    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "eval" / "manifests" / "bfcl_v4_study2.json")
        .read_text()
    )
    pairs = [(spec["category"], spec["role"]) for spec in manifest["files"]]
    assert len(pairs) == len(set(pairs))
