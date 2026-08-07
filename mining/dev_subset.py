"""Pin the study-2 development look-subset, deterministically and reproducibly.

Decision C (owner, #general msg 2244) adopts `live_multiple` as the development
set, because it presents 2-37 candidate tools per item and can therefore measure
the ranking skill study 2 trains -- which `live_simple`, at exactly one tool per
item, structurally cannot.

Scoring all 1,053 items at every look would cost roughly 4x the previous plan
(21,040 generations per arm at the look cap, against 5,160). The owner's rider
is therefore a pinned subset of 258 -- the size the abandoned `live_simple` plan
used, so per-look cost is unchanged and the prereg's spend table stays valid.

A subset invites exactly one question: was it chosen to produce a result? Three
properties answer it without anyone having to trust us.

- **The selection is a keyed hash order, not a shuffle.** Item order is
  `sha256(f"{seed}:{id}")` ascending. There is no RNG, no library version, and
  no language dependence: anyone can recompute the order with a hash function.
- **The seed is a fixed string, published here and in the receipt**, chosen
  before any model was run on any of these items.
- **The exclusion is a rule, not a list.** Items are dropped because they
  collide with a final scoring set, computed against the manifest at build time.
  The receipt records what the rule found rather than what we decided to omit.

Allocation is proportional by tool-count bucket with largest-remainder rounding,
so the subset preserves the population's ranking-pressure profile instead of
letting a hash order quietly over-sample two-tool items.

BFCL inputs are read-only and pinned (WORKING-AGREEMENT §5): the overlapping item
is excluded **in this receipt**, and no upstream file is edited.

Usage:
    python mining/dev_subset.py                 # build and write the receipt
    python mining/dev_subset.py --verify        # recompute; non-zero on mismatch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "eval" / "manifests" / "bfcl_v4_study2.json"
DEFAULT_RECEIPT = REPO_ROOT / "mining" / "receipts" / "study2_dev_look_subset.json"

CRITERION_ID = "study2-dev-look-subset/v1"
CRITERION = (
    "Population = the manifest-pinned live_multiple questions. Exclude any item whose "
    "question object is byte-identical, under canonical JSON, to a question object in a "
    "final scoring set (multiple, simple_python). Bucket the survivors by presented tool "
    "count into 2, 3, 4, 5-6, 7+; allocate 258 slots proportionally with largest-remainder "
    "rounding, ties broken by bucket order as listed; within each bucket take the first k "
    "ids by ascending sha256(seed + ':' + id). No RNG is used at any step."
)
SEED = "study2-dev-look-subset/v1:20260806"
SUBSET_SIZE = 258

DEV_CATEGORY = "live_multiple"
FINAL_CATEGORIES = ("multiple", "simple_python")

# The role string is machine-checkable and appears in two places on purpose: the
# manifest category and this receipt. A reader (or a test) that finds them
# disagreeing has found a real defect, which is the point of stating it twice.
DEV_ROLE = "development_selection_only"
BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("2", 2, 2),
    ("3", 3, 3),
    ("4", 4, 4),
    ("5-6", 5, 6),
    ("7+", 7, 10**9),
)


class SubsetError(RuntimeError):
    """The subset cannot be built as specified. Never a silent fallback."""


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_question(row: dict[str, Any]) -> str:
    return json.dumps(row["question"], sort_keys=True, separators=(",", ":"))


def _manifest_entry(manifest: dict[str, Any], category: str, role: str) -> dict[str, Any]:
    matches = [
        entry
        for entry in manifest["files"]
        if entry["category"] == category and entry["role"] == role
    ]
    if len(matches) != 1:
        raise SubsetError(f"expected exactly one {category}/{role} entry, found {len(matches)}")
    return matches[0]


def _verified_rows(manifest: dict[str, Any], category: str, role: str) -> list[dict[str, Any]]:
    """Read a pinned file, refusing it if it does not match its manifest pin."""
    entry = _manifest_entry(manifest, category, role)
    path = REPO_ROOT / entry["local_path"]
    payload = path.read_bytes()
    actual = _sha256(payload)
    if actual != entry["sha256"]:
        raise SubsetError(f"{entry['local_path']}: sha256 {actual} != pinned {entry['sha256']}")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if len(rows) != entry["row_count"]:
        raise SubsetError(f"{entry['local_path']}: {len(rows)} rows != pinned {entry['row_count']}")
    return rows


def _bucket_of(tool_count: int) -> str:
    for name, low, high in BUCKETS:
        if low <= tool_count <= high:
            return name
    raise SubsetError(f"tool count {tool_count} falls in no bucket")


def _allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Proportional allocation with largest-remainder rounding, ties by bucket order."""
    population = sum(sizes.values())
    if population < total:
        raise SubsetError(f"population {population} smaller than requested subset {total}")

    exact = {name: sizes[name] * total / population for name in sizes}
    floors = {name: int(value) for name, value in exact.items()}
    remaining = total - sum(floors.values())

    order = sorted(
        sizes,
        key=lambda name: (-(exact[name] - floors[name]), [b[0] for b in BUCKETS].index(name)),
    )
    for name in order[:remaining]:
        floors[name] += 1

    for name, allocated in floors.items():
        if allocated > sizes[name]:
            raise SubsetError(f"bucket {name}: allocated {allocated} > available {sizes[name]}")
    return floors


def build(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())

    declared_role = manifest["categories"][DEV_CATEGORY].get("study2_role")
    if declared_role != DEV_ROLE:
        raise SubsetError(
            f"manifest declares {DEV_CATEGORY} as {declared_role!r}, expected {DEV_ROLE!r} — "
            "refusing to pin a development subset from a category not marked as one"
        )

    dev_rows = _verified_rows(manifest, DEV_CATEGORY, "questions")
    _verified_rows(manifest, DEV_CATEGORY, "answer_key")  # must exist and match its pin

    final_questions: dict[str, str] = {}
    for category in FINAL_CATEGORIES:
        for row in _verified_rows(manifest, category, "questions"):
            final_questions[_canonical_question(row)] = f"{category}:{row['id']}"

    exclusions = []
    eligible = []
    for row in dev_rows:
        collision = final_questions.get(_canonical_question(row))
        if collision is not None:
            exclusions.append(
                {"id": row["id"], "reason": "question_collision_with_final_set",
                 "collides_with": collision}
            )
            continue
        eligible.append(row)

    by_bucket: dict[str, list[str]] = {name: [] for name, _, _ in BUCKETS}
    for row in eligible:
        by_bucket[_bucket_of(len(row["function"]))].append(row["id"])

    sizes = {name: len(ids) for name, ids in by_bucket.items()}
    allocation = _allocate(sizes, SUBSET_SIZE)

    selected: list[str] = []
    for name, _, _ in BUCKETS:
        ordered = sorted(by_bucket[name], key=lambda i: _sha256(f"{SEED}:{i}".encode()))
        selected.extend(ordered[: allocation[name]])

    if len(selected) != SUBSET_SIZE or len(set(selected)) != SUBSET_SIZE:
        raise SubsetError(f"selected {len(selected)} ids ({len(set(selected))} unique)")

    sorted_ids = sorted(selected)
    return {
        "criterion_id": CRITERION_ID,
        "criterion": CRITERION,
        "seed": SEED,
        "subset_size": SUBSET_SIZE,
        "implementation": {
            "path": "mining/dev_subset.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
        "manifest": {
            "path": str(manifest_path.relative_to(REPO_ROOT)),
            "sha256": _sha256(manifest_path.read_bytes()),
        },
        "source": {
            "category": DEV_CATEGORY,
            "sha256": _manifest_entry(manifest, DEV_CATEGORY, "questions")["sha256"],
            "answer_key_sha256": _manifest_entry(manifest, DEV_CATEGORY, "answer_key")["sha256"],
            "rows": len(dev_rows),
            "study2_role": manifest["categories"][DEV_CATEGORY]["study2_role"],
        },
        "study2_role": DEV_ROLE,
        "exclusions": exclusions,
        "eligible_rows": len(eligible),
        "buckets": [
            {"bucket": name, "eligible": sizes[name], "selected": allocation[name]}
            for name, _, _ in BUCKETS
        ],
        "selected_ids": sorted_ids,
        "sorted_id_sha256": _sha256(("\n".join(sorted_ids) + "\n").encode()),
        "endpoint_status": (
            "DEVELOPMENT ONLY. live_multiple selects checkpoints and therefore may never be "
            "reported as a study-2 endpoint (WORKING-AGREEMENT §6). The 795 items outside "
            "this subset are equally disqualified: the category as a whole is spent."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--verify", action="store_true",
                        help="Recompute and compare against the committed receipt.")
    args = parser.parse_args()

    built = build(args.manifest)

    if args.verify:
        if not args.receipt.exists():
            raise SubsetError(f"{args.receipt} does not exist")
        committed = json.loads(args.receipt.read_text())
        if committed != built:
            differing = sorted(
                key for key in set(committed) | set(built)
                if committed.get(key) != built.get(key)
            )
            print(f"MISMATCH in: {', '.join(differing)}", file=sys.stderr)
            raise SystemExit(1)
        print(f"verified {args.receipt.relative_to(REPO_ROOT)}: "
              f"{built['subset_size']} ids, sorted_id_sha256 {built['sorted_id_sha256']}")
        return

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(built, indent=2) + "\n")
    print(f"criterion  {built['criterion_id']}")
    print(f"seed       {built['seed']}")
    print(f"population {built['source']['rows']} rows, "
          f"{len(built['exclusions'])} excluded, {built['eligible_rows']} eligible")
    for bucket in built["buckets"]:
        print(f"  bucket {bucket['bucket']:>3}: {bucket['eligible']:>4} eligible "
              f"-> {bucket['selected']:>3} selected")
    print(f"selected   {built['subset_size']} ids, "
          f"sorted_id_sha256 {built['sorted_id_sha256']}")
    print(f"wrote {args.receipt.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
