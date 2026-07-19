"""Validate DPO preference pairs and emit a filtered, DPO-ready dataset.

Reads `data/processed/preferences.jsonl`, flags trivially-different pairs
(nothing for DPO to learn from), writes the surviving pairs to
`data/processed/preferences_dpo.jsonl`, and prints a summary report.

With `--sample N`, also emits a stratified-random spot-check sample
(across the 5 perturbation types) for manual review.

Usage
-----
    python data/validate_preferences.py
    python data/validate_preferences.py --sample 200

Input:  data/processed/preferences.jsonl
Output: data/processed/preferences_dpo.jsonl
        data/processed/spot_check_sample.jsonl        (when --sample given)
        data/processed/spot_check_sample.md           (when --sample given)
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

INPUT_PATH = Path("data/processed/preferences.jsonl")
OUTPUT_PATH = Path("data/processed/preferences_dpo.jsonl")
SAMPLE_JSONL = Path("data/processed/spot_check_sample.jsonl")
SAMPLE_MD = Path("data/processed/spot_check_sample.md")
SEED = 42

# Edit-distance ratio at or above this counts as "near-identical".
NEAR_IDENTICAL_RATIO = 0.98

PERTURBATIONS = [
    "wrong_tool_from_list",
    "hallucinated_tool",
    "missing_required_arg",
    "wrong_arg_value",
    "malformed_json",
]


# --------------------------------------------------------------------------
# Triviality signals
# --------------------------------------------------------------------------

def _canonicalize(text: str) -> Any:
    """Parse tool-call text and return a canonical (sorted-keys) structure.

    Returns None if unparsable — `malformed_json` rejections are expected
    to fail parse; those are non-trivial by construction.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _sort_keys(parsed)


def _sort_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_keys(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_sort_keys(x) for x in obj]
    return obj


def triviality_signals(chosen: str, rejected: str) -> dict[str, Any]:
    """Compute triviality signals and a single `is_trivial` verdict."""
    exact = chosen == rejected

    c_canon = _canonicalize(chosen)
    r_canon = _canonicalize(rejected)
    both_parse = c_canon is not None and r_canon is not None
    canon_equal = both_parse and c_canon == r_canon

    ratio = SequenceMatcher(None, chosen, rejected).ratio()
    near_identical = ratio >= NEAR_IDENTICAL_RATIO

    # An unparseable rejected side (chosen parses, rejected doesn't) is
    # semantically maximal difference even at tiny edit distance —
    # `malformed_json` perturbations are one-character edits by design.
    # Exempt such pairs from the edit-distance filter only.
    exempt_unparseable = c_canon is not None and r_canon is None

    # "Length-only" differences: parses match after canonicalization but
    # raw strings differ (whitespace/key-order/etc.). Nothing to learn.
    length_only = canon_equal and not exact

    trivial_reasons: list[str] = []
    if exact:
        trivial_reasons.append("exact_match")
    if canon_equal:
        trivial_reasons.append("canonical_json_equal")
    if length_only:
        trivial_reasons.append("length_or_whitespace_only")
    if near_identical and not exact and not exempt_unparseable:
        trivial_reasons.append("near_identical_edit_distance")

    return {
        "exact_match": exact,
        "canonical_equal": canon_equal,
        "edit_ratio": ratio,
        "near_identical": near_identical,
        "length_only_diff": length_only,
        "exempt_unparseable_rejected": exempt_unparseable,
        "trivial_reasons": trivial_reasons,
        "is_trivial": bool(trivial_reasons),
    }


# --------------------------------------------------------------------------
# Spot-check rendering
# --------------------------------------------------------------------------

def _extract_tools_block(system_content: str) -> str:
    """Return the tools section of the system prompt, verbatim if present."""
    marker = "Tools:"
    idx = system_content.find(marker)
    return system_content[idx:] if idx >= 0 else system_content


def render_pair_md(pair: dict, idx: int) -> str:
    system = ""
    user = ""
    for msg in pair.get("prompt_messages", []):
        if msg.get("role") == "system":
            system = msg.get("content", "")
        elif msg.get("role") == "user":
            user = msg.get("content", "")
    tools_block = _extract_tools_block(system)
    return (
        f"## Pair {idx} — {pair.get('perturbation_type', '?')} "
        f"({pair.get('source_id', '?')})\n\n"
        f"**User query:**\n\n> {user}\n\n"
        f"**Tools available:**\n\n```\n{tools_block}\n```\n\n"
        f"**Chosen (correct):**\n\n```json\n{pair.get('chosen', '')}\n```\n\n"
        f"**Rejected (perturbed):**\n\n```\n{pair.get('rejected', '')}\n```\n\n"
        f"**Verdict:** \n\n---\n\n"
    )


def write_spot_check(sample: list[dict]) -> None:
    with open(SAMPLE_JSONL, "w") as f:
        for p in sample:
            f.write(json.dumps(p) + "\n")
    with open(SAMPLE_MD, "w") as f:
        f.write(f"# Spot-check sample ({len(sample)} pairs)\n\n")
        f.write(
            "Stratified-random sample across perturbation types. "
            "Fill in `Verdict:` with `good` / `bad` / notes.\n\n"
        )
        for i, p in enumerate(sample, 1):
            f.write(render_pair_md(p, i))


def stratified_sample(
    pairs: list[dict], n: int, rng: random.Random
) -> list[dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_type[p.get("perturbation_type", "unknown")].append(p)

    types = [t for t in PERTURBATIONS if by_type.get(t)]
    per_bucket = n // len(types) if types else 0
    remainder = n - per_bucket * len(types)

    picks: list[dict] = []
    for t in types:
        bucket = by_type[t]
        take = min(len(bucket), per_bucket)
        picks.extend(rng.sample(bucket, take))

    # Distribute remainder from the largest buckets.
    if remainder > 0:
        leftovers = [p for t in types for p in by_type[t] if p not in picks]
        rng.shuffle(leftovers)
        picks.extend(leftovers[:remainder])

    rng.shuffle(picks)
    return picks


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If >0, write a stratified-random spot-check sample of this size.",
    )
    args = ap.parse_args()

    if not INPUT_PATH.exists():
        raise SystemExit(
            f"Input not found: {INPUT_PATH}\n"
            f"Run `python data/assemble_preferences.py` first."
        )

    pairs: list[dict] = []
    with open(INPUT_PATH) as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} pairs from {INPUT_PATH}")

    kept: list[dict] = []
    total_by_type: Counter[str] = Counter()
    filtered_by_reason: Counter[str] = Counter()
    filtered_by_type: Counter[str] = Counter()
    exempt_unparseable_count = 0

    for p in pairs:
        p_type = p.get("perturbation_type", "unknown")
        total_by_type[p_type] += 1
        sig = triviality_signals(p.get("chosen", ""), p.get("rejected", ""))
        if sig["exempt_unparseable_rejected"] and sig["near_identical"]:
            exempt_unparseable_count += 1
        if sig["is_trivial"]:
            for r in sig["trivial_reasons"]:
                filtered_by_reason[r] += 1
            filtered_by_type[p_type] += 1
            continue
        kept.append(p)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for p in kept:
            f.write(json.dumps(p) + "\n")

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"  Total in:          {len(pairs)}")
    print(f"  Filtered (trivial):{len(pairs) - len(kept)}")
    print(f"  Final DPO count:   {len(kept)}  (target >=10000)")

    print("\nCounts by perturbation type (in / filtered / kept):")
    for t in PERTURBATIONS:
        tin = total_by_type.get(t, 0)
        tf = filtered_by_type.get(t, 0)
        print(f"  {t:24s} {tin:6d} / {tf:5d} / {tin - tf:6d}")

    print("\nFiltered counts by reason (a pair can match multiple):")
    for reason, count in filtered_by_reason.most_common():
        print(f"  {reason:32s} {count}")
    print(f"  {'exempt_unparseable_rejected':32s} {exempt_unparseable_count}")

    if args.sample > 0:
        rng = random.Random(SEED)
        # Sample from the FULL set (pre-filter) so reviewers can also
        # catch trivial pairs that survived the automated filter.
        sample = stratified_sample(pairs, args.sample, rng)
        write_spot_check(sample)
        print(f"\nWrote spot-check sample ({len(sample)} pairs):")
        print(f"  {SAMPLE_JSONL}")
        print(f"  {SAMPLE_MD}")
        sample_type_counts = Counter(
            s.get("perturbation_type", "unknown") for s in sample
        )
        for t, c in sample_type_counts.most_common():
            print(f"    {t}: {c}")


if __name__ == "__main__":
    main()
