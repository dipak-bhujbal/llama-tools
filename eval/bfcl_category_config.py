"""Pure path configuration for supported single-call BFCL categories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CategoryPaths:
    questions: Path
    answer_key: Path
    default_output: Path


SUPPORTED_CATEGORIES = ("simple_python", "multiple")


def resolve_category_paths(repo_root: Path, category: str) -> CategoryPaths:
    if category not in SUPPORTED_CATEGORIES:
        choices = ", ".join(SUPPORTED_CATEGORIES)
        raise ValueError(f"unsupported BFCL category {category!r}; choose one of: {choices}")
    data_root = repo_root / "eval" / "bfcl_data"
    output_name = "bfcl_simple" if category == "simple_python" else f"bfcl_{category}"
    return CategoryPaths(
        questions=data_root / f"BFCL_v4_{category}.json",
        answer_key=data_root / "possible_answer" / f"BFCL_v4_{category}.json",
        default_output=repo_root / "eval" / "out" / output_name,
    )
