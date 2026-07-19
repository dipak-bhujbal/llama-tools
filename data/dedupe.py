"""Dedupe near-duplicate examples within the SFT set.

Uses MinHash + LSH (locality-sensitive hashing) to detect and drop
near-duplicates based on the user-query text. Preserves the first
occurrence of each duplicate group.

Why user-query-only?
    Two examples with different tool responses but the same user query
    still teach the model the same input distribution. If we kept both,
    we'd double-count that query's training signal for no learning gain.
    Deduping on the query catches this without over-aggressively removing
    legitimate variants that share tool call structure.

Why MinHash + LSH (not exact hash)?
    Public datasets accumulate near-duplicates: "get the weather in NYC"
    vs "get me the weather in NYC" vs "please get weather in NYC" are
    effectively the same query. MinHash + LSH is a sublinear
    algorithm for near-duplicate detection at scale that catches these
    variants (Jaccard similarity threshold, tunable).

BFCL leakage dedup
------------------
DEFERRED to Week 7 (concurrent with eval harness setup). Rationale: BFCL v3's
HF data format shifts as new task categories are added; wiring dedup against
it now would likely be wrong and give a false sense of safety. Week 7's
`eval/bfcl_v3.py` will import a shared query-set, and this dedupe.py will
be extended to consume it once the eval harness lands.

Usage
-----
    python data/dedupe.py

Input:  data/processed/sft.jsonl
Output: data/processed/sft_dedup.jsonl
"""

import json
import re
from pathlib import Path

from datasketch import MinHash, MinHashLSH

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

INPUT_PATH = Path("data/processed/sft.jsonl")
OUTPUT_PATH = Path("data/processed/sft_dedup.jsonl")

# MinHash config. 128 permutations is the industry-standard default for
# near-duplicate detection; more perms = better precision but slower.
MINHASH_NUM_PERM = 128

# Jaccard similarity threshold. 0.7 is conservative — catches obvious
# paraphrases without merging legitimate distinct queries. Tunable if we
# see too many false-positive drops.
JACCARD_THRESHOLD = 0.7

# Word-level 5-grams for the tokenizer. 5 is a common choice: captures
# enough context to distinguish real duplicates from coincidental
# vocabulary overlap.
NGRAM_SIZE = 5


# --------------------------------------------------------------------------
# Tokenization + MinHash
# --------------------------------------------------------------------------

def tokenize_ngrams(text: str, n: int = NGRAM_SIZE) -> set[str]:
    """Return the set of word-level n-grams from `text`, lowercased.

    Small texts (< n words) return single-word "n-grams" as a fallback so
    MinHash still has something to hash.
    """
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return set(words)
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def make_minhash(text: str, num_perm: int = MINHASH_NUM_PERM) -> MinHash:
    """Compute a MinHash signature of the tokenized text."""
    m = MinHash(num_perm=num_perm)
    for token in tokenize_ngrams(text):
        m.update(token.encode("utf-8"))
    return m


def extract_user_query(example: dict) -> str:
    """Return the first user-role message content from an example."""
    for msg in example.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "") or ""
    return ""


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(
            f"Input not found: {INPUT_PATH}\n"
            f"Run `python data/assemble_sft.py` first."
        )

    # Load
    examples: list[dict] = []
    with open(INPUT_PATH) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples from {INPUT_PATH}")

    # Build LSH index incrementally: for each example, check if a
    # near-duplicate has already been indexed; if yes, drop; if no, index.
    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    kept: list[dict] = []
    dropped = 0
    empty_query = 0

    for i, ex in enumerate(examples):
        query = extract_user_query(ex)
        if not query.strip():
            empty_query += 1
            continue
        m = make_minhash(query)
        if lsh.query(m):  # near-duplicate exists in index
            dropped += 1
            continue
        lsh.insert(f"kept-{len(kept)}", m)
        kept.append(ex)

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")

    # Per-source breakdown after dedup
    source_counts: dict[str, int] = {}
    for ex in kept:
        src = ex.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"  Input:                  {len(examples)}")
    print(f"  Kept:                   {len(kept)}")
    print(f"  Dropped (near-dup):     {dropped}")
    print(f"  Dropped (empty query):  {empty_query}")
    print("  Source breakdown (kept):")
    for src, count in sorted(source_counts.items()):
        print(f"    {src}: {count}")
    print(
        "\nNext step: build preference pairs "
        "(`python data/assemble_preferences.py` — coming next commit)."
    )


if __name__ == "__main__":
    main()
