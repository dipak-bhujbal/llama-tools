"""Pure path configuration for supported single-call BFCL categories.

`live_multiple` is here because owner Decision C (#general msg 2244) made it the
study-2 development set. It differs from the scoring categories in one way that
must not be forgettable: **only the 258 IDs pinned in the committed subset receipt
are ever scored**, and it may never be reported as a study endpoint. That is
carried on the category itself (`is_development_subset`) rather than left to a
caller to remember.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CategoryPaths:
    questions: Path
    answer_key: Path
    default_output: Path
    # True only for the development set: the runner must restrict the run to the
    # pinned subset (eval/dev_subset_gate.py) and refuse to score the full file.
    is_development_subset: bool = False


SUPPORTED_CATEGORIES = ("simple_python", "multiple", "live_multiple")

# Categories that may be reported as study-2 endpoints. `live_multiple` is
# deliberately absent: selecting checkpoints on it spends the whole parent
# category (prereg §3.2), and the manifest labels it `development_selection_only`.
SCORING_CATEGORIES = ("simple_python", "multiple")

DEVELOPMENT_CATEGORY = "live_multiple"


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
        is_development_subset=category == DEVELOPMENT_CATEGORY,
    )
