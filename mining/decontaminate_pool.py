"""Screen the mining pool against the frozen eval sets, and emit the artifact.

This is the producer prereg A2.2 has been waiting on: the Phase 2 gate reads a
committed decontamination artifact, and until now `mining/decontaminate.py` was
a screening library with no way to write one.

The ordering is prereg §2.9 and is load-bearing. Eligibility runs first, so the
screen input is the *retained* population and not the raw file:

    cleaned source - prompt-ineligible - target-structural exclusions = screen input

Criterion `bfcl-pool-decontamination/v1` pins the cascade: 13-gram user-text
overlap first, exact presented-function-name collision second, stop at the first
match. Shared keys resolve to the first category loaded from the manifest, so
manifest order is part of the criterion and is recorded.

Weights leave here as the integer triple `(n_multi, n_single, N)`. Decimals are
derived at display time, never stored, so `P_std` is computed from exact ratios.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mining.decontaminate import Decontaminator  # noqa: E402
from mining.pool_strata import (  # noqa: E402
    ELIGIBILITY_CRITERION_ID,
    INELIGIBLE,
    MULTI,
    SINGLE,
    _system_prompt,
    classify_target,
    presented_names,
    stratum_of,
)

CRITERION_ID = "bfcl-pool-decontamination/v1"
CRITERION = (
    "Screen each retained pool prompt against every manifest entry with role=questions. "
    "Cascade, in this order, stopping at the first match: (1) 13-gram overlap on "
    "normalized user text; (2) exact collision between a presented function name and a "
    "name presented by an eval item. Where two categories share a gram or a name, the "
    "first category loaded from the manifest wins, so manifest order is part of this "
    "criterion. A retained row that cannot be screened deterministically is a hard "
    "failure, never a new exclusion bucket."
)


class DecontaminationError(RuntimeError):
    """A retained row could not be screened, or carries a name defect."""


DEFAULT_PREFLIGHT = REPO_ROOT / "mining" / "receipts" / "sft_dedup_v2_target_preflight.json"


def _relative(path: Path) -> str:
    """Repo-relative, whether the caller passed an absolute or relative path."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _id_digest(ids: list[str]) -> str:
    return _digest(("\n".join(sorted(ids)) + "\n").encode())


def _verify_preflight(preflight_path: Path, pool_sha: str) -> dict[str, Any]:
    """The eligibility receipt must be about *this* pool, and must have passed.

    Binding only the criterion id would let an artifact cite an eligibility rule
    whose receipt was computed against different bytes, or that failed.
    """
    raw = preflight_path.read_bytes()
    receipt = json.loads(raw)
    if receipt.get("sha256") != pool_sha:
        raise DecontaminationError(
            f"preflight receipt is about {receipt.get('sha256')}, not this pool {pool_sha}"
        )
    if receipt.get("criterion_id") != ELIGIBILITY_CRITERION_ID:
        raise DecontaminationError(
            f"preflight criterion {receipt.get('criterion_id')!r} != {ELIGIBILITY_CRITERION_ID!r}"
        )
    # `is True`, not truthiness: a fail-closed gate must not be opened by the
    # string "false", or by 1, or by any other value that happens to be truthy.
    if receipt.get("passed") is not True:
        raise DecontaminationError(
            f"preflight receipt records passed={receipt.get('passed')!r}, which is not True; "
            f"refusing to screen"
        )
    return {
        "path": _relative(preflight_path),
        "sha256": _digest(raw),
        "criterion_id": receipt["criterion_id"],
        "counts": {k: receipt[k] for k in
                   ("raw_rows", "prompt_ineligible", "structurally_excluded", "retained_rows")},
    }


def build_artifact(
    pool_path: Path,
    manifest_path: Path,
    preflight_path: Path = DEFAULT_PREFLIGHT,
) -> dict[str, Any]:
    payload = pool_path.read_bytes()
    pool_sha = _digest(payload)
    preflight = _verify_preflight(preflight_path, pool_sha)
    screener = Decontaminator([manifest_path])

    eligible: list[tuple[str, str, str, list[str]]] = []
    prompt_ineligible = structurally_excluded = 0

    for index, line in enumerate(payload.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        system = _system_prompt(row)
        stratum, _ = stratum_of(system)
        if stratum == INELIGIBLE:
            prompt_ineligible += 1
            continue
        assistant = next(
            (m.get("content", "") for m in row.get("messages", []) if m.get("role") == "assistant"),
            "",
        )
        kind, called = classify_target(assistant)
        if kind == "unreadable":
            structurally_excluded += 1
            continue
        presented = presented_names(system)
        # A name defect is never an exclusion. Discarding the parsed names here
        # let a retained row calling a tool its prompt never presented pass
        # straight through into the survivor set.
        missing = sorted(called - presented)
        if missing:
            raise DecontaminationError(
                f"{row.get('source_id')}: retained row calls {missing}, which its prompt does "
                f"not present; a name defect is a defect, not an exclusion"
            )
        user_text = " ".join(
            m.get("content", "") for m in row.get("messages", []) if m.get("role") == "user"
        )
        row_id = str(row.get("source_id") or f"line:{index + 1}")
        eligible.append((row_id, stratum, user_text, sorted(presented)))

    survivors: dict[str, int] = {MULTI: 0, SINGLE: 0}
    dropped: dict[str, int] = {MULTI: 0, SINGLE: 0}
    drop_reasons: dict[str, int] = {}
    survivor_ids: list[str] = []
    input_ids: list[str] = []

    for row_id, stratum, user_text, names in eligible:
        input_ids.append(row_id)
        try:
            hit, reason = screener.is_contaminated(user_text, names)
        except Exception as exc:  # a retained row we cannot screen is a hard failure
            raise DecontaminationError(
                f"{row_id}: survived eligibility but could not be screened: {exc}"
            ) from exc
        if hit:
            dropped[stratum] += 1
            key = str(reason).split(":")[0]
            drop_reasons[key] = drop_reasons.get(key, 0) + 1
        else:
            survivors[stratum] += 1
            survivor_ids.append(row_id)

    # The receipt's headline reconciliation must agree with what we just
    # re-derived. Embedding the counts without comparing them let an artifact
    # cite a receipt describing a different population.
    derived = {
        "raw_rows": prompt_ineligible + structurally_excluded + len(eligible),
        "prompt_ineligible": prompt_ineligible,
        "structurally_excluded": structurally_excluded,
        "retained_rows": len(eligible),
    }
    disagreements = {
        field: {"receipt": preflight["counts"][field], "derived": value}
        for field, value in derived.items()
        if preflight["counts"][field] != value
    }
    if disagreements:
        raise DecontaminationError(
            f"preflight receipt disagrees with the re-derived population: {disagreements}"
        )

    n_multi, n_single = survivors[MULTI], survivors[SINGLE]
    total = n_multi + n_single
    screen_input = len(eligible)

    pre = {MULTI: 0, SINGLE: 0}
    for _rid, stratum, _t, _n in eligible:
        pre[stratum] += 1

    return {
        # Ordered deliberately: the reconciliation is stated before any drop or
        # survival count, per §2.9. Written with sort_keys=False for that reason.
        "criterion_id": CRITERION_ID,
        "criterion": CRITERION,
        "implementation": [
            {"path": rel, "sha256": _digest((REPO_ROOT / rel).read_bytes())}
            for rel in ("mining/decontaminate_pool.py", "mining/decontaminate.py",
                        "mining/pool_strata.py")
        ],
        "eligibility_criterion_id": ELIGIBILITY_CRITERION_ID,
        "eligibility_receipt": preflight,
        "source": {"path": _relative(pool_path), "sha256": pool_sha},
        "manifest": {"path": _relative(manifest_path),
                     "sha256": _digest(manifest_path.read_bytes())},
        "screened_question_files": screener.screened_manifest(),
        "reconciliation": (
            f"{prompt_ineligible + structurally_excluded + screen_input} cleaned source "
            f"- {prompt_ineligible} prompt-ineligible - {structurally_excluded} "
            f"target-structural exclusions = {screen_input} screen inputs"
        ),
        "screen_input_rows": screen_input,
        "pre_screen": {"multi": pre[MULTI], "single": pre[SINGLE], "total": screen_input},
        "screen_input_id_sha256": _id_digest(input_ids),
        "prompt_ineligible": prompt_ineligible,
        "structurally_excluded": structurally_excluded,
        "dropped": {"multi": dropped[MULTI], "single": dropped[SINGLE],
                    "total": dropped[MULTI] + dropped[SINGLE], "by_reason": drop_reasons},
        "survivors": {"multi": n_multi, "single": n_single, "total": total},
        "post_screen_id_sha256": _id_digest(survivor_ids),
        # Integer triple only. Any decimal is derived at display time.
        "weights": {"n_multi": n_multi, "n_single": n_single, "N": total},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pool", type=Path)
    parser.add_argument("--manifest", type=Path,
                        default=REPO_ROOT / "eval" / "manifests" / "bfcl_v4_study2.json")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    args = parser.parse_args()

    art = build_artifact(args.pool, args.manifest, args.preflight)
    print(f"criterion   {art['criterion_id']}")
    print(f"source      {art['source']['path']}  sha256={art['source']['sha256'][:16]}…")
    print(f"screened    {len(art['screened_question_files'])} question files")
    print(f"{art['reconciliation']}")
    print(f"dropped     {art['dropped']['total']}  {art['dropped']['by_reason']}")
    w = art["weights"]
    share = w["n_multi"] / w["N"] if w["N"] else 0.0
    print(f"survivors   multi={w['n_multi']} single={w['n_single']} N={w['N']}")
    print(f"weights     ({w['n_multi']}, {w['n_single']}, {w['N']})   "
          f"multi share {share * 100:.3f}% (derived, not stored)")
    if args.json:
        args.json.write_text(json.dumps(art, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
