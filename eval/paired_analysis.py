"""Paired per-item comparison between candidates scored on the same items.

Every candidate in a sweep sees the identical prompt set, so their per-item
outcomes are strongly correlated. Comparing marginal accuracies — or checking
whether two marginal binomial CIs overlap — is not a valid test of whether two
candidates differ: it ignores the pairing and systematically understates power.

The correct test conditions on the items where the two candidates *disagree*
(the discordant pairs) and asks whether the disagreements are lopsided. That is
exactly McNemar's test, computed here exactly (two-sided binomial, p=0.5)
rather than with the chi-square approximation, because discordant counts here
are small enough that the approximation is unreliable.

Usage:
    python eval/paired_analysis.py --generations eval/results/study1_bfcl_simple_generations.jsonl
    python eval/paired_analysis.py --generations <path> --reference sft
"""

import argparse
import json
from math import comb
from pathlib import Path


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    b = items the reference got right and the candidate got wrong
    c = items the candidate got right and the reference got wrong
    Concordant items carry no information about a difference and drop out.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(max(b, c), n + 1))
    return min(1.0, 2 * tail / 2**n)


def load_outcomes(path: Path, field: str = "overall_ok") -> dict:
    """Return {candidate: {item_id: bool}} from a generations.jsonl."""
    by_candidate: dict = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_candidate.setdefault(row["model_name"], {})[row["id"]] = bool(row[field])
    return by_candidate


def compare(by_candidate: dict, reference: str) -> list:
    """Paired comparison of every candidate against `reference`."""
    if reference not in by_candidate:
        raise SystemExit(
            f"reference {reference!r} not in generations "
            f"(found: {sorted(by_candidate)})"
        )
    ref = by_candidate[reference]
    results = []
    for name, outcomes in by_candidate.items():
        if name == reference:
            continue
        shared = sorted(set(ref) & set(outcomes))
        if len(shared) != len(ref) or len(shared) != len(outcomes):
            raise SystemExit(
                f"{name} and {reference} were not scored on the same items "
                f"({len(outcomes)} vs {len(ref)}, {len(shared)} shared) — "
                f"a paired test requires identical item sets"
            )
        b = sum(1 for i in shared if ref[i] and not outcomes[i])
        c = sum(1 for i in shared if not ref[i] and outcomes[i])
        results.append(
            {
                "candidate": name,
                "reference": reference,
                "n_items": len(shared),
                "reference_correct": sum(ref.values()),
                "candidate_correct": sum(outcomes.values()),
                "marginal_delta": sum(outcomes.values()) - sum(ref.values()),
                "reference_only_correct": b,
                "candidate_only_correct": c,
                "discordant": b + c,
                "p_exact_mcnemar": exact_mcnemar_p(b, c),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--reference", default="sft",
                        help="Candidate every other is compared against")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional JSON output path")
    args = parser.parse_args()

    by_candidate = load_outcomes(args.generations)
    results = compare(by_candidate, args.reference)

    header = (
        f"{'candidate':16s} {'marg':>6s} {'ref-only':>9s} "
        f"{'cand-only':>10s} {'p':>9s}  verdict"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        verdict = (
            "differs" if r["p_exact_mcnemar"] < args.alpha
            else "not distinguishable"
        )
        print(
            f"{r['candidate']:16s} {r['marginal_delta']:+6d} "
            f"{r['reference_only_correct']:9d} {r['candidate_only_correct']:10d} "
            f"{r['p_exact_mcnemar']:9.4f}  {verdict}"
        )

    if args.out:
        args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
