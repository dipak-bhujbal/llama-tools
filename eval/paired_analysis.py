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
import math
from math import comb
from pathlib import Path
from statistics import NormalDist


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


def holm_adjust(p_values: list) -> list:
    """Holm step-down adjusted p-values, in the input order.

    Testing several candidates against one reference is a family. Reporting the
    smallest raw p-value as significant without correcting for how many
    contrasts were run inflates the familywise error rate — which is exactly
    how a marginal result gets over-claimed.
    """
    indexed = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(indexed):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)  # enforce monotonicity
        adjusted[idx] = running
    return adjusted


def _constrained_p21(b: int, c: int, n: int, delta: float) -> float:
    """Constrained ML estimate of `p21` under H0: difference = `delta`.

    The positive root of `2n·q² + B·q + C = 0` with
    `B = -b - c + (2n - c + b)·delta` and `C = -b·delta·(1 - delta)`.
    """
    a = 2 * n
    quad_b = -b - c + (2 * n - c + b) * delta
    quad_c = -b * delta * (1.0 - delta)
    discriminant = quad_b * quad_b - 4 * a * quad_c
    return (math.sqrt(max(discriminant, 0.0)) - quad_b) / (2 * a)


def tango_score(b: int, c: int, n: int, delta: float) -> float:
    """Tango's score statistic for H0: (candidate - reference) difference = `delta`.

    Positive when the candidate leads the null value, negative when it trails, and
    monotonically decreasing in `delta` — which is what makes the interval below a
    simple bracketed root-find rather than a search.
    """
    variance = n * (2 * _constrained_p21(b, c, n, delta) + delta * (1.0 - delta))
    if variance <= 0:
        return math.inf if (c - b - n * delta) > 0 else -math.inf
    return (c - b - n * delta) / math.sqrt(variance)


def tango_interval(b: int, c: int, n: int, conf_level: float = 0.95,
                   tolerance: float = 1e-12) -> tuple[float, float]:
    """Tango's score confidence interval for the paired difference of proportions.

    **Direction convention, stated because sign errors here are silent:** `b` is
    the count where the *reference* is correct and the candidate is not, `c` the
    reverse, matching `exact_mcnemar_p`. The interval is therefore for
    **candidate - reference**, and its point estimate is `(c - b) / n`.

    Method: Tango (1998), *Statistics in Medicine* 17:891-908 — the interval is
    the set of `delta` the score test does not reject, `{delta : |z(delta)| <= z_crit}`.
    Its endpoints are found here by bisection on `tango_score`, which is
    monotone decreasing, to a tolerance far tighter than any reported figure.

    Degenerate cases follow the method's documented behaviour: with `b == n` every
    discordant item favours the reference, so the lower endpoint is the boundary
    value -1; with `c == n`, the upper endpoint is +1.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if b < 0 or c < 0:
        raise ValueError(f"b and c must be non-negative, got b={b}, c={c}")
    if b + c > n:
        raise ValueError(f"discordant count {b + c} exceeds n={n}")
    if not 0.0 < conf_level < 1.0:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")

    z_crit = NormalDist().inv_cdf(1.0 - (1.0 - conf_level) / 2.0)
    point = (c - b) / n

    def _root(low: float, high: float, target: float) -> float:
        """Bisect for `tango_score == target` on a bracket where it is monotone."""
        for _ in range(200):
            mid = 0.5 * (low + high)
            if tango_score(b, c, n, mid) > target:
                low = mid
            else:
                high = mid
            if high - low < tolerance:
                break
        return 0.5 * (low + high)

    lower = -1.0 if b == n else _root(-1.0, point, z_crit)
    upper = 1.0 if c == n else _root(point, 1.0, -z_crit)
    return lower, upper


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
                "tango_ci_95": list(tango_interval(b, c, len(shared))),
            }
        )
    for row, adjusted in zip(
        results, holm_adjust([r["p_exact_mcnemar"] for r in results]), strict=True
    ):
        row["p_holm_adjusted"] = adjusted
        row["family_size"] = len(results)
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
        f"{'cand-only':>10s} {'p_raw':>8s} {'p_holm':>8s} "
        f"{'95% Tango CI':>22s}  verdict"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        # The verdict keys off the ADJUSTED p-value. Several candidates are
        # being compared against one reference, so the raw p-value of the
        # best-looking contrast is not the familywise error rate.
        verdict = (
            "differs" if r["p_holm_adjusted"] < args.alpha
            else "not distinguishable"
        )
        low, high = r["tango_ci_95"]
        interval = f"[{100 * low:+7.2f}, {100 * high:+7.2f}]pp"
        print(
            f"{r['candidate']:16s} {r['marginal_delta']:+6d} "
            f"{r['reference_only_correct']:9d} {r['candidate_only_correct']:10d} "
            f"{r['p_exact_mcnemar']:8.4f} {r['p_holm_adjusted']:8.4f} "
            f"{interval:>22s}  {verdict}"
        )
    print(
        f"\nfamily size = {len(results)} contrasts against '{args.reference}'; "
        f"verdicts use Holm-adjusted p at alpha={args.alpha}."
    )
    print(
        "Tango score intervals are 95% and for candidate - reference, in "
        "percentage points; reported regardless of direction or significance "
        "(prereg A1.4), and never Holm-adjusted — an interval is not a test."
    )

    if args.out:
        args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
