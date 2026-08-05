"""Compare the two candidate BFCL v4 answer keys and re-score study 1 under each.

Phase -1 froze the eval inputs by hash and caught that two upstream revisions
carry different answer keys for `simple_python`:

- `58f57e91…` — the BFCL v4 *release* commit.
- `9d8416a9…` — a later upstream *data-fix* revision, and the key the
  2026-07-20 study-1 run actually scored against. This is what the manifest
  currently pins.

They differ at exactly one row, `simple_python_363`. This script re-scores the
committed per-item generations under both, and checks the property that makes
the choice safe for the DPO kill decision: whether the disputed item is
concordant across candidates (all right or all wrong), since concordant items
contribute nothing to a paired comparison.

It also measures the scorer-normalization question that item 363 is a symptom
of: how often answer keys use module-qualified function names, and how often a
`multiple` row's tools are distinguishable *only* by that module prefix.

Run:
    python eval/answer_key_comparison.py
    python eval/answer_key_comparison.py --json eval/results/answer_key_comparison.json

Inputs are the committed, manifest-pinned files; nothing is fetched and no
model is called.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "eval" / "bfcl_data"
GENERATIONS = REPO_ROOT / "eval" / "results" / "study1_bfcl_simple_generations.jsonl"

DISPUTED_ID = "simple_python_363"
RELEASE_COMMIT = "58f57e9124ea981403792dd51e00a6577e621fae"
DATAFIX_COMMIT = "9d8416a96d1d69975493f1b6d60ff07d12a1726a"
# The release key's value at the disputed row. The release-commit file is not
# committed to this repo, so this is a stated input, not a hash-verified one --
# see the "provenance" note in the emitted report.
RELEASE_NAME_AT_DISPUTED = "find_closest"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _by_id(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in _load_jsonl(path)}


def key_names(answer_row: dict) -> set[str]:
    names: set[str] = set()
    for entry in answer_row["ground_truth"]:
        names |= set(entry.keys())
    return names


def score_tables() -> dict:
    """Per-candidate totals under the pinned key and under the release key."""
    rows = _load_jsonl(GENERATIONS)
    totals: Counter[str] = Counter()
    pinned: Counter[str] = Counter()
    for row in rows:
        totals[row["model_name"]] += 1
        pinned[row["model_name"]] += bool(row["overall_ok"])

    disputed = {row["model_name"]: row for row in rows if row["id"] == DISPUTED_ID}
    release: dict[str, int] = {}
    emitted: dict[str, str] = {}
    for model in totals:
        row = disputed[model]
        emitted[model] = row["parsed_name"]
        # Under the release key the item is correct only if the model emitted
        # the unqualified name AND its arguments were already accepted.
        would_pass = row["parsed_name"] == RELEASE_NAME_AT_DISPUTED and row["args_ok"]
        release[model] = pinned[model] - int(bool(row["overall_ok"])) + int(would_pass)

    return {
        "n": dict(totals),
        "pinned_key": dict(pinned),
        "release_key": release,
        "disputed_item_emitted": emitted,
        "disputed_item_concordant": len(set(emitted.values())) == 1,
    }


def qualified_name_stats() -> dict:
    """How exposed each category is to the qualified-vs-unqualified question."""
    out: dict[str, dict] = {}
    for category in ("simple_python", "multiple", "live_simple"):
        questions = _by_id(DATA / f"BFCL_v4_{category}.json")
        answers = _by_id(DATA / "possible_answer" / f"BFCL_v4_{category}.json")

        qualified_rows = 0
        key_not_presented = 0
        tail_collisions = 0
        for row_id, question in questions.items():
            presented = {fn["name"] for fn in question["function"]}
            keyed = key_names(answers[row_id])
            if any("." in name for name in keyed):
                qualified_rows += 1
            if not keyed <= presented:
                key_not_presented += 1
            tails = [name.split(".")[-1] for name in presented]
            if len(set(tails)) < len(tails):
                tail_collisions += 1

        out[category] = {
            "rows": len(questions),
            "rows_with_module_qualified_key": qualified_rows,
            "rows_where_key_name_not_among_presented_tools": key_not_presented,
            "rows_where_tools_share_an_unqualified_tail": tail_collisions,
        }
    return out


def disputed_item_detail() -> dict:
    questions = _by_id(DATA / "BFCL_v4_simple_python.json")
    answers = _by_id(DATA / "possible_answer" / "BFCL_v4_simple_python.json")
    return {
        "id": DISPUTED_ID,
        "tools_presented_to_the_model": [
            fn["name"] for fn in questions[DISPUTED_ID]["function"]
        ],
        "pinned_key_expects": sorted(key_names(answers[DISPUTED_ID])),
        "release_key_expects": [RELEASE_NAME_AT_DISPUTED],
    }


def build_report() -> dict:
    return {
        "schema_version": 1,
        "release_commit": RELEASE_COMMIT,
        "datafix_commit": DATAFIX_COMMIT,
        "disputed_item": disputed_item_detail(),
        "scores": score_tables(),
        "qualified_name_exposure": qualified_name_stats(),
        "provenance": {
            "generations": str(GENERATIONS.relative_to(REPO_ROOT)),
            "note": (
                "The release-commit answer-key file is NOT committed to this repo. "
                "Its value at the disputed row is a stated input reproduced from an "
                "independent check, not a hash-verified one. Pinning the release key "
                "canonically would require committing that file and adding it to "
                "eval/manifests/bfcl_v4_study2.json with its hashes."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    args = parser.parse_args()

    report = build_report()
    scores = report["scores"]
    detail = report["disputed_item"]

    print(f"disputed item: {detail['id']}")
    print(f"  tools presented to the model : {detail['tools_presented_to_the_model']}")
    print(f"  pinned key ({DATAFIX_COMMIT[:8]}) expects : {detail['pinned_key_expects']}")
    print(f"  release key ({RELEASE_COMMIT[:8]}) expects: {detail['release_key_expects']}")
    print()

    print(f"{'candidate':<10} {'pinned key':>12} {'release key':>13}   emitted at disputed item")
    for model in sorted(scores["n"]):
        n = scores["n"][model]
        print(
            f"{model:<10} {scores['pinned_key'][model]:>8}/{n} "
            f"{scores['release_key'][model]:>9}/{n}   "
            f"{scores['disputed_item_emitted'][model]}"
        )
    print()
    print(f"disputed item concordant across candidates: {scores['disputed_item_concordant']}")
    print("  (concordant items contribute nothing to a paired test, so every")
    print("   pairwise delta and McNemar discordant set is invariant to the key)")
    print()

    print("qualified-name exposure by category:")
    print(f"  {'category':<14} {'rows':>5} {'qualified key':>14} "
          f"{'key not offered':>16} {'tail collisions':>16}")
    for category, stats in report["qualified_name_exposure"].items():
        rows = stats["rows"]
        print(
            f"  {category:<14} {rows:>5} "
            f"{stats['rows_with_module_qualified_key']:>9} "
            f"({stats['rows_with_module_qualified_key'] / rows:>4.0%}) "
            f"{stats['rows_where_key_name_not_among_presented_tools']:>16} "
            f"{stats['rows_where_tools_share_an_unqualified_tail']:>11} "
            f"({stats['rows_where_tools_share_an_unqualified_tail'] / rows:>4.0%})"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
