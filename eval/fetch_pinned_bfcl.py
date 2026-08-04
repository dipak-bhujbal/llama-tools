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


class VerificationError(ValueError):
    """The bytes on disk or from upstream do not match the frozen manifest."""


def _digest(name: str, payload: bytes) -> str:
    return hashlib.new(name, payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324 - Git object id


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
        (candidate for candidate in certificate_candidates if candidate and Path(candidate).is_file()),
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
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise VerificationError(f"unsupported or empty manifest: {path}")
    return manifest


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
        verified.append(destination)
        if backup_dir is not None:
            _content_addressed_backup(destination, backup_dir, spec["sha256"])
        print(
            f"verified {spec['category']}/{spec['role']}: "
            f"{spec['row_count']} rows sha256={spec['sha256']} revision={revision}"
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
