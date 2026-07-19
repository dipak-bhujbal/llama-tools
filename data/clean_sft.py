"""Drop SFT examples whose assistant turn contains the xLAM Python-expr bug.

v2 cleanup (2026-07-19)
-----------------------
A spot-check audit today surfaced an upstream annotation bug in the xLAM
source: some argument values are written as Python expressions
(e.g. `"[0.02] * 5"`) instead of literal JSON arrays. Any example whose
ASSISTANT turn contains one teaches the model to emit non-JSON — corrupt
training signal, not a valid tool call. The DPO pipeline already filters
these via `PYTHON_EXPR_IN_CHOSEN` in `data/validate_preferences.py`; this
script applies the same filter to the SFT set so the two can't drift apart.

Only assistant-turn (training-target) matches are dropped. User/system
turns that mention such expressions are left alone — they're not what the
model learns to produce.

Usage
-----
    python data/clean_sft.py

Input:  data/processed/sft_dedup.jsonl
Output: data/processed/sft_dedup_v2.jsonl
"""

import json
from pathlib import Path

from validate_preferences import PYTHON_EXPR_IN_CHOSEN

INPUT_PATH = Path("data/processed/sft_dedup.jsonl")
OUTPUT_PATH = Path("data/processed/sft_dedup_v2.jsonl")


def has_python_expr_in_assistant(example: dict) -> bool:
    """Return True if any assistant-turn content matches the bug pattern."""
    for msg in example.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        if PYTHON_EXPR_IN_CHOSEN.search(msg.get("content", "")):
            return True
    return False


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(
            f"Input not found: {INPUT_PATH}\n"
            f"Run `python data/assemble_sft.py` and `python data/dedupe.py` first."
        )

    examples: list[dict] = []
    with open(INPUT_PATH) as f:
        for line in f:
            examples.append(json.loads(line))

    kept: list[dict] = []
    dropped_source_ids: list[str] = []

    for ex in examples:
        if has_python_expr_in_assistant(ex):
            dropped_source_ids.append(str(ex.get("source_id", "?")))
            continue
        kept.append(ex)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {OUTPUT_PATH}")
    print(f"  Total in:      {len(examples)}")
    print(f"  Dropped:       {len(dropped_source_ids)}")
    print(f"  Total out:     {len(kept)}")
    print("\nDropped source_ids:")
    for sid in dropped_source_ids:
        print(f"  {sid}")


if __name__ == "__main__":
    main()
