"""DPO v2 Stage 2: assemble hard preference pairs from sampled failures.

Implements ADR-007. Reads `data/processed/failure_candidates.jsonl` (output
of data/sample_failures.py), applies the guards, and emits the v2 training
set plus a stats report.

Guards:
- Drop rows whose rejected is canonically equal to chosen (defensive —
  scoring in Stage 1 should already have excluded successes).
- Keep unparseable rejecteds unconditionally (ADR-005 exemption rationale:
  textual similarity is irrelevant when the semantic difference is maximal).
- Dedupe canonically-identical rejecteds per prompt; cap 2 pairs per prompt.
- Excluded source ids (shared with validate_preferences.py) never pass.

Gate (pre-registered in ADR-007): if the final pair count is below 1,500 the
run prints ABORT and exits non-zero — DPO v2 training must not be launched.

Usage:
    python data/build_dpo_v2_pairs.py

Outputs:
    data/processed/preferences_dpo_v2.jsonl
    data/processed/dpo_v2_stats.md
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_preferences import EXCLUDED_SOURCE_IDS, _canonicalize  # noqa: E402

IN_PATH = Path("data/processed/failure_candidates.jsonl")
OUT_PATH = Path("data/processed/preferences_dpo_v2.jsonl")
STATS_PATH = Path("data/processed/dpo_v2_stats.md")

MIN_PAIRS = 1500          # ADR-007 abort floor
MAX_PAIRS_PER_PROMPT = 2


def rejected_key(text: str) -> str:
    canon = _canonicalize(text)
    if canon is not None:
        return json.dumps(canon, sort_keys=True)
    return text.strip()


def main() -> None:
    if not IN_PATH.exists():
        sys.exit(f"Missing {IN_PATH} — run data/sample_failures.py first.")

    per_prompt: dict[str, list[dict]] = defaultdict(list)
    sampled_prompts = 0
    dropped = Counter()

    with IN_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            if "_sampled_source_id" in row:
                sampled_prompts += 1
                continue
            if row["source_id"] in EXCLUDED_SOURCE_IDS:
                dropped["excluded_source_id"] += 1
                continue
            c_canon = _canonicalize(row["chosen"])
            r_canon = _canonicalize(row["rejected"])
            if c_canon is not None and r_canon is not None and c_canon == r_canon:
                dropped["rejected_equals_chosen"] += 1
                continue
            per_prompt[row["source_id"]].append(row)

    pairs: list[dict] = []
    for source_id, rows in per_prompt.items():
        seen: set[str] = set()
        kept_here = 0
        # Prefer the most severe failure types first when capping.
        severity = {"invalid_json": 0, "wrong_tool": 1, "wrong_args": 2}
        for row in sorted(rows, key=lambda r: severity.get(r["failure_type"], 3)):
            key = rejected_key(row["rejected"])
            if key in seen:
                dropped["duplicate_rejected"] += 1
                continue
            seen.add(key)
            if kept_here >= MAX_PAIRS_PER_PROMPT:
                dropped["per_prompt_cap"] += 1
                continue
            kept_here += 1
            pairs.append(row)

    by_type = Counter(p["failure_type"] for p in pairs)
    prompts_with_failure = len(per_prompt)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    lines = [
        "# DPO v2 pair-assembly stats",
        "",
        f"- Prompts sampled (Stage 1): {sampled_prompts}",
        f"- Prompts with >=1 failure: {prompts_with_failure}"
        + (f" ({prompts_with_failure / sampled_prompts:.1%})" if sampled_prompts else ""),
        f"- **Final pairs: {len(pairs)}** (floor: {MIN_PAIRS})",
        "",
        "| failure type | pairs |",
        "|---|---|",
    ]
    lines += [f"| {t} | {n} |" for t, n in by_type.most_common()]
    lines += ["", "Dropped:", ""]
    lines += [f"- {k}: {v}" for k, v in dropped.most_common()] or ["- none"]
    STATS_PATH.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote {OUT_PATH} and {STATS_PATH}")

    if len(pairs) < MIN_PAIRS:
        print(f"\nABORT: {len(pairs)} pairs < {MIN_PAIRS} floor (ADR-007). "
              "Do NOT launch train/dpo_v2_full.py — ship SFT and stop.")
        sys.exit(1)
    print("\nGate PASSED — cleared to launch train/dpo_v2_full.py")


if __name__ == "__main__":
    main()
