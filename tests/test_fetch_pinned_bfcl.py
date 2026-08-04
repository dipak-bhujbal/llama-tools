from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval.fetch_pinned_bfcl import VerificationError, _git_blob_sha1, verify_payload


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
    assert manifest["schema_version"] == 1
    assert manifest["default_source_revision"] == (
        "9d8416a96d1d69975493f1b6d60ff07d12a1726a"
    )
    assert len(manifest["files"]) == 7
    assert all("source_revision" in spec for spec in manifest["files"])
