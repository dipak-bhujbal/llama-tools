"""Fetch or verify the exact BFCL v4 files frozen for llama-tools study 2.

The manifest pins an immutable upstream Git commit, Git blob ids, SHA-256
digests, row counts, unique-id counts, and the sorted id-set digest. Files are
verified before an atomic write. Raw benchmark data remains gitignored; only
the acquisition code and manifest are committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "eval" / "manifests" / "bfcl_v4_study2.json"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{revision}/{path}"

MANIFEST_SCHEMA_VERSION = 2
QUESTIONS_ROLE = "questions"
ANSWER_KEY_ROLE = "answer_key"

# This module is imported both as a script-relative sibling (`fetch_pinned_bfcl`,
# by eval/*.py) and as a namespace-package module (`eval.fetch_pinned_bfcl`, by
# the tests). The scoring rule has to come in under either name.
try:
    from bfcl_scoring import KeyDefectError, preflight_key_names
except ImportError:  # pragma: no cover - exercised by whichever import form is not used
    from eval.bfcl_scoring import KeyDefectError, preflight_key_names


class VerificationError(ValueError):
    """The bytes on disk or from upstream do not match the frozen manifest."""


def _digest(name: str, payload: bytes) -> str:
    return hashlib.new(name, payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()  # Git object id, not a security digest


def _jsonl_ids(payload: bytes) -> list[str]:
    ids: list[str] = []
    for line_number, raw_line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerificationError(f"invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(row, dict) or "id" not in row:
            raise VerificationError(f"line {line_number} has no top-level id")
        ids.append(str(row["id"]))
    return ids


def verify_payload(payload: bytes, spec: dict[str, Any]) -> None:
    actual_sha256 = _digest("sha256", payload)
    if actual_sha256 != spec["sha256"]:
        raise VerificationError(
            f"{spec['local_path']}: sha256 {actual_sha256} != {spec['sha256']}"
        )
    actual_blob = _git_blob_sha1(payload)
    if actual_blob != spec["git_blob_sha1"]:
        raise VerificationError(
            f"{spec['local_path']}: git blob {actual_blob} != {spec['git_blob_sha1']}"
        )

    ids = _jsonl_ids(payload)
    if len(ids) != spec["row_count"]:
        raise VerificationError(
            f"{spec['local_path']}: {len(ids)} rows != {spec['row_count']}"
        )
    unique_ids = set(ids)
    if len(unique_ids) != spec["unique_id_count"]:
        raise VerificationError(
            f"{spec['local_path']}: {len(unique_ids)} unique ids "
            f"!= {spec['unique_id_count']}"
        )
    sorted_ids = ("\n".join(sorted(ids)) + "\n").encode()
    actual_id_digest = _digest("sha256", sorted_ids)
    if actual_id_digest != spec["sorted_id_sha256"]:
        raise VerificationError(
            f"{spec['local_path']}: sorted id digest {actual_id_digest} "
            f"!= {spec['sorted_id_sha256']}"
        )


def _safe_local_path(destination_root: Path, relative_path: str) -> Path:
    root = destination_root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise VerificationError(f"manifest path escapes destination root: {relative_path}")
    return candidate


def _download(url: str) -> bytes:
    verify_paths = ssl.get_default_verify_paths()
    certificate_candidates = [
        os.environ.get("SSL_CERT_FILE"),
        verify_paths.cafile,
        "/etc/ssl/cert.pem",
    ]
    certificate_file = next(
        (
            candidate
            for candidate in certificate_candidates
            if candidate and Path(candidate).is_file()
        ),
        None,
    )
    context = ssl.create_default_context(cafile=certificate_file)
    request = urllib.request.Request(
        url, headers={"User-Agent": "llama-tools-pinned-bfcl-fetch/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        return response.read()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _content_addressed_backup(source: Path, backup_root: Path, sha256: str) -> Path:
    destination = backup_root / "sha256" / sha256
    if destination.exists():
        if _digest("sha256", destination.read_bytes()) != sha256:
            raise VerificationError(f"backup collision or corruption: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _digest("sha256", destination.read_bytes()) != sha256:
        destination.unlink(missing_ok=True)
        raise VerificationError(f"backup verification failed: {destination}")
    return destination


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    version = manifest.get("schema_version")
    if version == 1:
        raise VerificationError(
            f"{path}: schema_version 1 predates the standing answer-name preflight and "
            f"declares no per-category answer-key policy, so a category with no key would "
            f"be skipped rather than checked; re-pin at schema_version {MANIFEST_SCHEMA_VERSION}"
        )
    if version != MANIFEST_SCHEMA_VERSION or not manifest.get("files"):
        raise VerificationError(f"unsupported or empty manifest: {path}")
    _reject_duplicate_roles(manifest, path)
    return manifest


def _reject_duplicate_roles(manifest: dict[str, Any], path: Path | str = "manifest") -> None:
    """One (category, role) may name exactly one file.

    Everything downstream — verification, the preflight, the comparison's input
    resolution — keys files by (category, role) and would take the last one
    written. Two answer keys for a category then means the first is pinned but
    never checked: a defective key followed by a clean one verifies clean and
    the standing preflight reports a pass it did not earn. Rejecting the
    ambiguity here is the only place that covers every consumer at once.
    """
    seen: dict[tuple[str, str], str] = {}
    for spec in manifest["files"]:
        key = (spec["category"], spec["role"])
        if key in seen:
            raise VerificationError(
                f"{path}: {spec['category']}/{spec['role']} is pinned twice "
                f"({seen[key]} and {spec['local_path']}); one (category, role) names one file, "
                f"because every consumer resolves by that pair and would silently take one"
            )
        seen[key] = spec["local_path"]


def _rows_by_id(payload: bytes, label: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in payload.decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["id"])] = row
    if not rows:
        raise VerificationError(f"{label}: no rows")
    return rows


def preflight_manifest_keys(
    manifest: dict[str, Any],
    payloads: dict[tuple[str, str], bytes],
) -> list[dict[str, Any]]:
    """Check every pinned answer key against the tools its items actually present.

    `simple_python_363` was found by hand, one category at a time. This runs the
    same rule over every category the manifest pins, every time the manifest is
    verified, so its cousins surface before a study freezes rather than after it
    publishes.

    Every category is either checked or explicitly declared keyless; a category
    the manifest never classifies is an error, because "no answer key found" and
    "no answer key exists by schema" are indistinguishable from the file list
    alone and only one of them is safe to pass over.
    """
    declared = manifest.get("categories")
    if not declared:
        raise VerificationError(
            "manifest declares no `categories` block, so answer-key policy per category "
            "is unknown; refusing to guess which categories may legitimately lack a key"
        )
    # Re-checked here rather than trusted from load_manifest: this is the
    # function that makes the "every pinned answer key was checked" claim, so it
    # must not be able to make it about a file list it never disambiguated.
    _reject_duplicate_roles(manifest)

    keys_by_category: dict[str, dict[str, Any]] = {}
    comparison_only: list[dict[str, Any]] = []
    categories_seen: set[str] = set()
    for spec in manifest["files"]:
        categories_seen.add(spec["category"])
        if spec["role"] == ANSWER_KEY_ROLE:
            keys_by_category[spec["category"]] = spec
        elif spec["role"] not in (QUESTIONS_ROLE, ANSWER_KEY_ROLE):
            comparison_only.append(spec)

    unclassified = sorted(categories_seen - set(declared))
    if unclassified:
        raise VerificationError(
            f"manifest pins files for {unclassified} but declares no answer-key policy for "
            f"them; add each to `categories` as `required` or `none_by_schema`"
        )
    unpinned = sorted(set(declared) - categories_seen)
    if unpinned:
        raise VerificationError(
            f"manifest declares policy for {unpinned} but pins no files for them"
        )

    receipts: list[dict[str, Any]] = []
    for category in sorted(declared):
        policy = declared[category].get("answer_key_policy")
        key_spec = keys_by_category.get(category)
        if policy == "required":
            if key_spec is None:
                raise VerificationError(
                    f"{category}: answer_key_policy is `required` but the manifest pins no "
                    f"{ANSWER_KEY_ROLE} for it"
                )
            receipts.append(_preflight_one(category, key_spec, payloads, expect="clean"))
        elif policy == "none_by_schema":
            if key_spec is not None:
                raise VerificationError(
                    f"{category}: answer_key_policy is `none_by_schema` but the manifest "
                    f"pins {key_spec['local_path']}; one of the two is wrong"
                )
            receipts.append(
                {
                    "category": category,
                    "role": None,
                    "status": "no-answer-key-by-schema",
                    "items_checked": 0,
                    "note": declared[category].get("note", ""),
                }
            )
        else:
            raise VerificationError(
                f"{category}: unknown answer_key_policy {policy!r}; expected `required` or "
                f"`none_by_schema`"
            )

    for spec in comparison_only:
        expectation = spec.get("preflight_expectation")
        if expectation not in ("clean", "defective"):
            raise VerificationError(
                f"{spec['category']}/{spec['role']}: comparison-only key declares "
                f"preflight_expectation {expectation!r}; expected `clean` or `defective`"
            )
        receipts.append(_preflight_one(spec["category"], spec, payloads, expect=expectation))

    return receipts


def _preflight_one(
    category: str,
    key_spec: dict[str, Any],
    payloads: dict[tuple[str, str], bytes],
    *,
    expect: str,
) -> dict[str, Any]:
    """Run the answer-name preflight for one key and hold it to `expect`."""
    question_payload = payloads.get((category, QUESTIONS_ROLE))
    if question_payload is None:
        raise VerificationError(
            f"{category}/{key_spec['role']}: cannot preflight an answer key without the "
            f"category's pinned questions file"
        )
    questions = _rows_by_id(question_payload, f"{category}/{QUESTIONS_ROLE}")
    answers = _rows_by_id(payloads[(category, key_spec["role"])], f"{category}/{key_spec['role']}")

    defect: str | None = None
    try:
        checked = preflight_key_names(questions, answers)
    except KeyDefectError as exc:
        checked, defect = len(answers), str(exc)

    if expect == "clean" and defect is not None:
        raise VerificationError(f"{category}/{key_spec['role']}: {defect}")
    if expect == "defective" and defect is None:
        raise VerificationError(
            f"{category}/{key_spec['role']}: manifest records this key as defective, but it "
            f"now passes the answer-name preflight; the canonical-key adjudication rested on "
            f"it failing and must be revisited rather than left standing"
        )
    return {
        "category": category,
        "role": key_spec["role"],
        "status": "defective-as-recorded" if defect else "clean",
        "items_checked": checked,
        "defect": defect,
    }


def process_manifest(
    manifest_path: Path,
    destination_root: Path,
    *,
    verify_only: bool,
    backup_dir: Path | None,
) -> list[Path]:
    manifest = load_manifest(manifest_path)
    default_revision = manifest["default_source_revision"]
    repository = manifest["upstream_repository"]
    verified: list[Path] = []
    payloads: dict[tuple[str, str], bytes] = {}

    for spec in manifest["files"]:
        revision = spec.get("source_revision", default_revision)
        destination = _safe_local_path(destination_root, spec["local_path"])
        if verify_only:
            if not destination.is_file():
                raise VerificationError(f"missing frozen file: {destination}")
            payload = destination.read_bytes()
        else:
            url = RAW_URL.format(
                repo=repository, revision=revision, path=spec["upstream_path"]
            )
            payload = _download(url)
            verify_payload(payload, spec)
            if not destination.exists() or destination.read_bytes() != payload:
                _atomic_write(destination, payload)

        verify_payload(payload, spec)
        payloads[(spec["category"], spec["role"])] = payload
        verified.append(destination)
        if backup_dir is not None:
            _content_addressed_backup(destination, backup_dir, spec["sha256"])
        print(
            f"verified {spec['category']}/{spec['role']}: "
            f"{spec['row_count']} rows sha256={spec['sha256']} revision={revision}"
        )

    # Bytes matching the pin only proves the file is the one we froze. The
    # preflight is what says the thing we froze is scoreable at all.
    for receipt in preflight_manifest_keys(manifest, payloads):
        if receipt["status"] == "no-answer-key-by-schema":
            print(f"preflight {receipt['category']}: no answer key by schema (declared)")
        else:
            print(
                f"preflight {receipt['category']}/{receipt['role']}: {receipt['status']} "
                f"({receipt['items_checked']} key items checked)"
            )
    return verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=REPO_ROOT,
        help="Root under which manifest local_path values are resolved.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not use the network or write data; verify existing cached files.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Optional human-controlled directory for content-addressed backup copies.",
    )
    args = parser.parse_args()
    process_manifest(
        args.manifest,
        args.destination_root,
        verify_only=args.verify_only,
        backup_dir=args.backup_dir,
    )


if __name__ == "__main__":
    main()
