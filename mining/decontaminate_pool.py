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
    """A row survived eligibility but could not be screened."""


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _id_digest(ids: list[str]) -> str:
    return _digest(("\n".join(sorted(ids)) + "\n").encode())


def build_artifact(pool_path: Path, manifest_path: Path) -> dict[str, Any]:
    payload = pool_path.read_bytes()
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
        kind, _names = classify_target(assistant)
        if kind == "unreadable":
            structurally_excluded += 1
            continue
        user_text = " ".join(
            m.get("content", "") for m in row.get("messages", []) if m.get("role") == "user"
        )
        row_id = str(row.get("source_id") or f"line:{index + 1}")
        eligible.append((row_id, stratum, user_text, sorted(presented_names(system))))

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

    n_multi, n_single = survivors[MULTI], survivors[SINGLE]
    total = n_multi + n_single
    screen_input = len(eligible)

    return {
        "criterion_id": CRITERION_ID,
        "criterion": CRITERION,
        "implementation": "mining/decontaminate_pool.py + mining/decontaminate.py",
        "eligibility_criterion_id": ELIGIBILITY_CRITERION_ID,
        "source": {"path": pool_path.name, "sha256": _digest(payload)},
        "manifest": {"path": str(manifest_path.relative_to(REPO_ROOT)),
                     "sha256": _digest(manifest_path.read_bytes())},
        "screened_question_files": screener.screened_manifest(),
        "reconciliation": (
            f"{prompt_ineligible + structurally_excluded + screen_input} cleaned source "
            f"- {prompt_ineligible} prompt-ineligible - {structurally_excluded} "
            f"target-structural exclusions = {screen_input} screen inputs"
        ),
        "screen_input_rows": screen_input,
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
    args = parser.parse_args()

    art = build_artifact(args.pool, args.manifest)
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
        args.json.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
