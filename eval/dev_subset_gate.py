"""Fail-closed loader for the study-2 development look subset (prereg §3.2, §3.3).

Decision C made `live_multiple` the development set, but only the 258 IDs pinned in
`mining/receipts/study2_dev_look_subset.json` are ever scored. Every kill line and
every eligibility comparison is denominated in items against that exact set, so a
run that scored 257 items, or 258 slightly different ones, would silently move
every threshold that referenced the frozen baseline.

This module is the gate that makes that impossible to do by accident. It refuses
unless **all** of the following hold:

- the committed receipt's bytes match the digest pinned here;
- the manifest digest the receipt recorded matches the manifest on disk;
- the pinned `live_multiple` question and answer-key files match their manifest
  digests;
- the manifest still labels the category `development_selection_only`;
- the receipt's ID set is exactly 258 unique IDs, and every one exists in the
  questions file.

**Division of labour with the tests, stated so neither is mistaken for the other.**
This module enforces *pins*; `tests/test_dev_slice_preflight.py` proves the receipt
*reproduces* from `mining/dev_subset.py` under its published seed. A pin catches
drift; regeneration catches a receipt that was rewritten together with its pin.
Runtime needs the first and must not depend on the mining package to start.

Nothing here runs a model or scores anything. It hands back an ID set, or raises.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

DEV_CATEGORY = "live_multiple"
DEV_ROLE = "development_selection_only"
DEV_SUBSET_SIZE = 258

# Pinned by prereg §3.2 / Amendment 3 and by the preflight test constants.
SUBSET_RECEIPT_SHA256 = "5a9510711adee429b8d0b2d7e20b35cb57278d052f39cb19d33f86a46b57b33b"
SUBSET_SORTED_ID_SHA256 = "a91d8271224d7a50f68c27c0070b114173412c2591ba304ac7a6048506760b64"


class DevSubsetGateError(RuntimeError):
    """The development subset cannot be trusted, so nothing may be scored on it."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_entry(manifest: dict, category: str, role: str) -> dict:
    matches = [
        entry
        for entry in manifest["files"]
        if entry["category"] == category and entry["role"] == role
    ]
    if len(matches) != 1:
        raise DevSubsetGateError(
            f"expected exactly one {category}/{role} manifest entry, found {len(matches)}"
        )
    return matches[0]


def load_dev_subset_ids(repo_root: Path) -> list[str]:
    """Return the 258 pinned development IDs, or raise `DevSubsetGateError`."""
    receipt_path = repo_root / "mining" / "receipts" / "study2_dev_look_subset.json"
    manifest_path = repo_root / "eval" / "manifests" / "bfcl_v4_study2.json"

    if not receipt_path.exists():
        raise DevSubsetGateError(
            f"{receipt_path} is missing; the development subset is not pinned and "
            "nothing may be scored on it"
        )

    receipt_bytes = receipt_path.read_bytes()
    actual_receipt = _sha256(receipt_bytes)
    if actual_receipt != SUBSET_RECEIPT_SHA256:
        raise DevSubsetGateError(
            f"subset receipt sha256 {actual_receipt} != pinned {SUBSET_RECEIPT_SHA256}"
        )

    receipt = json.loads(receipt_bytes)
    manifest_bytes = manifest_path.read_bytes()
    actual_manifest = _sha256(manifest_bytes)
    if receipt["manifest"]["sha256"] != actual_manifest:
        raise DevSubsetGateError(
            f"receipt was built against manifest {receipt['manifest']['sha256']}, "
            f"but the manifest on disk is {actual_manifest}"
        )

    manifest = json.loads(manifest_bytes)
    declared_role = manifest["categories"].get(DEV_CATEGORY, {}).get("study2_role")
    if declared_role != DEV_ROLE:
        raise DevSubsetGateError(
            f"manifest labels {DEV_CATEGORY} as {declared_role!r}, not {DEV_ROLE!r}"
        )

    for role in ("questions", "answer_key"):
        entry = _manifest_entry(manifest, DEV_CATEGORY, role)
        path = repo_root / entry["local_path"]
        actual = _sha256(path.read_bytes())
        if actual != entry["sha256"]:
            raise DevSubsetGateError(
                f"{entry['local_path']}: sha256 {actual} != pinned {entry['sha256']}"
            )

    ids = list(receipt["selected_ids"])
    if len(ids) != DEV_SUBSET_SIZE or len(set(ids)) != DEV_SUBSET_SIZE:
        raise DevSubsetGateError(
            f"receipt carries {len(ids)} IDs ({len(set(ids))} unique), "
            f"expected {DEV_SUBSET_SIZE} unique"
        )

    digest = _sha256(("\n".join(sorted(ids)) + "\n").encode())
    if digest != SUBSET_SORTED_ID_SHA256:
        raise DevSubsetGateError(
            f"sorted-ID digest {digest} != pinned {SUBSET_SORTED_ID_SHA256}"
        )

    questions_path = repo_root / _manifest_entry(manifest, DEV_CATEGORY, "questions")["local_path"]
    available = {
        json.loads(line)["id"]
        for line in questions_path.read_text().splitlines()
        if line.strip()
    }
    missing = sorted(set(ids) - available)
    if missing:
        raise DevSubsetGateError(
            f"{len(missing)} pinned development IDs are absent from "
            f"{questions_path.name}, first: {missing[:3]}"
        )

    return ids


def restrict_to_dev_subset(
    prompt_rows: list[dict],
    answer_rows: list[dict],
    subset_ids: list[str],
) -> tuple[list[dict], list[dict]]:
    """Restrict *both* loaded files to exactly the pinned IDs, or raise.

    Prereg §3.2 requires the development runner to load exactly the receipt's 258
    ids, **match the same 258 answer rows**, and refuse missing, duplicate,
    excluded, or extra ids. Filtering the questions alone does not satisfy that:
    the answer key still carries all 1,053 parent rows, so the two sides would
    disagree on size and the run would either abort or — worse, in a future
    refactor that reconciles by intersection — score a set nobody pinned.

    Extra ids cannot survive here because the returned rows are a subset of
    `subset_ids` by construction, and the *excluded* collision item is already
    kept out by the sorted-ID digest checked in `load_dev_subset_ids`.
    """
    wanted = set(subset_ids)
    if len(wanted) != len(subset_ids):
        duplicated = sorted({i for i, n in Counter(subset_ids).items() if n > 1})
        raise DevSubsetGateError(
            f"pinned ID list carries duplicates, first: {duplicated[:3]}"
        )

    restricted: dict[str, list[dict]] = {}
    for label, rows in (("question", prompt_rows), ("answer", answer_rows)):
        kept = [row for row in rows if row["id"] in wanted]
        counts = Counter(row["id"] for row in kept)
        duplicated = sorted(i for i, n in counts.items() if n > 1)
        if duplicated:
            raise DevSubsetGateError(
                f"{label} file carries duplicate pinned ids, first: {duplicated[:3]}"
            )
        absent = sorted(wanted - set(counts))
        if absent:
            raise DevSubsetGateError(
                f"{len(absent)} pinned ids have no {label} row, first: {absent[:3]}"
            )
        restricted[label] = kept

    return restricted["question"], restricted["answer"]
