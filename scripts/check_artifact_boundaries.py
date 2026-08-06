"""Fail closed on study-2 fixture, eval-data, and legacy-artifact boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

LEGACY_FILENAMES = {
    "scale_pairs.py",
    "dpo_pairs_train.jsonl",
    "dpo_pairs_eval.jsonl",
    "audit_sample_50.jsonl",
    "DPO_pairs_data.zip",
}

TEXT_SUFFIXES = {".json", ".jsonl", ".py", ".toml", ".yaml", ".yml"}


def _source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for relative_root in ("train", "data", "mining"):
        source_root = root / relative_root
        if not source_root.exists():
            continue
        files.extend(
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix in TEXT_SUFFIXES
        )
    return sorted(files)


def check_tree(root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []

    for path in root.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.name in LEGACY_FILENAMES:
            errors.append(f"legacy fabricated-meta artifact present: {path.relative_to(root)}")
        if path.suffix == ".jsonl":
            try:
                if "exact_match_checker_v2" in path.read_text(errors="replace"):
                    errors.append(
                        f"fabricated verifier provenance present: {path.relative_to(root)}"
                    )
            except OSError as exc:
                errors.append(f"could not inspect {path.relative_to(root)}: {exc}")

    for path in _source_files(root):
        relative = path.relative_to(root)
        text = path.read_text(errors="replace")
        if relative.parts[0] in {"train", "data"} and "tests/fixtures" in text:
            errors.append(f"training/data source references tests/fixtures: {relative}")
        if relative.parts[0] == "train" and "eval/bfcl_data" in text:
            errors.append(f"training source references held-out eval data: {relative}")

    return errors


def main() -> None:
    errors = check_tree()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("artifact boundaries: OK")


if __name__ == "__main__":
    main()
