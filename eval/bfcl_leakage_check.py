"""BFCL v3 train/eval leakage check.

Pulled forward from Week 7. Before we ever publish BFCL v3 scores, we need to
know whether our training data overlaps the BFCL eval questions.

Method
------
Two independent signals, run against BOTH training sets
(data/processed/sft_dedup.jsonl and data/processed/preferences_dpo.jsonl):

1. **MinHash + LSH** (near-duplicate) — reuses the exact tokenization
   (word 5-grams, lowercased) and Jaccard threshold (0.7) from
   `data/dedupe.py` so results are directly comparable to the Week 3 internal
   dedup.
2. **Exact normalized match** (stricter) — whitespace-collapsed + lowercased
   equality on the user-query text. A hit here is a smoking gun.

BFCL source
-----------
Berkeley Function Calling Leaderboard v3 lives in
`ShishirPatil/gorilla` at `berkeley-function-call-leaderboard/bfcl_eval/data/`.
Each category ships as a JSONL file `BFCL_v3_<category>.json` — every line is
one task with an `id` and (usually) a `question` field that is a list of
conversation turns. We download **every** `BFCL_v3_*.json` we can find, then
extract the user-turn text from each task and check every extracted query.

Cache: files land in `eval/bfcl_data/` (gitignored). Re-run is a no-op if
files already present unless `--refresh` is passed.

Usage
-----
    python eval/bfcl_leakage_check.py                # download (if needed) + check + write report
    python eval/bfcl_leakage_check.py --refresh      # force re-download of BFCL files
    python eval/bfcl_leakage_check.py --offline      # skip download; use whatever is cached
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from datasketch import MinHash, MinHashLSH

# ---------------------------------------------------------------------------
# Paths + config (kept in sync with data/dedupe.py)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SFT_PATH = REPO_ROOT / "data" / "processed" / "sft_dedup.jsonl"
DPO_PATH = REPO_ROOT / "data" / "processed" / "preferences_dpo.jsonl"

BFCL_CACHE_DIR = REPO_ROOT / "eval" / "bfcl_data"
REPORT_PATH = REPO_ROOT / "eval" / "bfcl_leakage_report.md"

# Match Week 3 dedup params so signals are directly comparable.
MINHASH_NUM_PERM = 128
JACCARD_THRESHOLD = 0.7
NGRAM_SIZE = 5

# GitHub source of truth for BFCL v3 question files.
BFCL_REPO = "ShishirPatil/gorilla"
BFCL_DATA_DIR = "berkeley-function-call-leaderboard/bfcl_eval/data"
GITHUB_API = f"https://api.github.com/repos/{BFCL_REPO}/contents/{BFCL_DATA_DIR}"
RAW_BASE = (
    f"https://raw.githubusercontent.com/{BFCL_REPO}/main/{BFCL_DATA_DIR}"
)

# Fallback list (in case GitHub API is rate-limited). Covers the categories
# documented on the BFCL leaderboard as of v3. If a file isn't present at the
# raw URL, we skip it gracefully.
FALLBACK_FILES = [
    "BFCL_v3_simple.json",
    "BFCL_v3_multiple.json",
    "BFCL_v3_parallel.json",
    "BFCL_v3_parallel_multiple.json",
    "BFCL_v3_java.json",
    "BFCL_v3_javascript.json",
    "BFCL_v3_irrelevance.json",
    "BFCL_v3_live_simple.json",
    "BFCL_v3_live_multiple.json",
    "BFCL_v3_live_parallel.json",
    "BFCL_v3_live_parallel_multiple.json",
    "BFCL_v3_live_irrelevance.json",
    "BFCL_v3_live_relevance.json",
    "BFCL_v3_multi_turn_base.json",
    "BFCL_v3_multi_turn_miss_func.json",
    "BFCL_v3_multi_turn_miss_param.json",
    "BFCL_v3_multi_turn_long_context.json",
]


# ---------------------------------------------------------------------------
# Shingling + MinHash (identical to data/dedupe.py)
# ---------------------------------------------------------------------------

def tokenize_ngrams(text: str, n: int = NGRAM_SIZE) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return set(words)
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def make_minhash(text: str, num_perm: int = MINHASH_NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for token in tokenize_ngrams(text):
        m.update(token.encode("utf-8"))
    return m


def normalize_exact(text: str) -> str:
    """Whitespace-collapse + lowercase for the exact-match signal."""
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Download BFCL question files
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "llama-tools-bfcl-leakage-check/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def list_bfcl_files() -> list[str]:
    """Ask GitHub API for the current file list; fall back to a hardcoded list."""
    try:
        data = json.loads(_http_get(GITHUB_API).decode("utf-8"))
        names = [
            item["name"]
            for item in data
            if isinstance(item, dict)
            and item.get("name", "").startswith("BFCL_v3_")
            and item.get("name", "").endswith(".json")
        ]
        if names:
            return sorted(names)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as e:
        print(f"[warn] GitHub API listing failed ({e}); using fallback file list.")
    return list(FALLBACK_FILES)


def download_bfcl(refresh: bool = False, offline: bool = False) -> list[Path]:
    """Ensure BFCL v3 question files are in BFCL_CACHE_DIR. Return their paths."""
    BFCL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if offline:
        # Upstream moved to v4 (2026-07); accept any cached version.
        cached = sorted(BFCL_CACHE_DIR.glob("BFCL_v*_*.json"))
        if not cached:
            raise SystemExit(
                f"--offline set but no cached files in {BFCL_CACHE_DIR}. "
                f"Run once without --offline first."
            )
        return cached

    files = list_bfcl_files()
    print(f"BFCL file candidates: {len(files)}")
    got: list[Path] = []
    for name in files:
        dest = BFCL_CACHE_DIR / name
        if dest.exists() and not refresh:
            got.append(dest)
            continue
        url = f"{RAW_BASE}/{name}"
        try:
            payload = _http_get(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # File name from fallback list may not exist in current tree.
                continue
            print(f"[warn] {name}: HTTP {e.code}")
            continue
        except urllib.error.URLError as e:
            print(f"[warn] {name}: {e}")
            continue
        dest.write_bytes(payload)
        got.append(dest)
        time.sleep(0.05)  # be polite to raw.githubusercontent.com
    print(f"Downloaded/cached: {len(got)} files in {BFCL_CACHE_DIR}")
    return got


# ---------------------------------------------------------------------------
# Extract queries from BFCL tasks
# ---------------------------------------------------------------------------

def extract_bfcl_queries(path: Path) -> list[tuple[str, str, str]]:
    """Return (bfcl_id, turn_key, user_text) tuples from one BFCL file.

    BFCL tasks are JSONL, one task per line. The `question` field is
    conventionally a list-of-lists of message dicts (outer list = one entry
    per multi-turn round, inner = messages for that round). We extract
    every user-role message text across all rounds.
    """
    out: list[tuple[str, str, str]] = []
    with open(path) as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = str(task.get("id", f"{path.stem}#{lineno}"))
            question = task.get("question")
            texts: list[tuple[str, str]] = []

            def _collect_from_msg(msg, tag):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        texts.append((tag, content))

            if isinstance(question, list):
                for i, round_or_msg in enumerate(question):
                    if isinstance(round_or_msg, list):
                        for j, msg in enumerate(round_or_msg):
                            _collect_from_msg(msg, f"round{i}.msg{j}")
                    else:
                        _collect_from_msg(round_or_msg, f"msg{i}")
            elif isinstance(question, str) and question.strip():
                texts.append(("question", question))

            # Some BFCL variants use a flat top-level user field.
            for alt in ("user_query", "prompt"):
                v = task.get(alt)
                if isinstance(v, str) and v.strip():
                    texts.append((alt, v))

            for tag, txt in texts:
                out.append((task_id, tag, txt))
    return out


# ---------------------------------------------------------------------------
# Extract queries from our training sets
# ---------------------------------------------------------------------------

def extract_sft_queries(path: Path) -> list[tuple[str, str]]:
    """Return (source_id, user_query) for each SFT row."""
    rows: list[tuple[str, str]] = []
    with open(path) as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            sid = str(ex.get("id") or ex.get("source_id") or f"sft#{i}")
            for msg in ex.get("messages", []):
                if msg.get("role") == "user":
                    content = msg.get("content", "") or ""
                    if content.strip():
                        rows.append((sid, content))
                    break
    return rows


def extract_dpo_queries(path: Path) -> list[tuple[str, str]]:
    """Return (source_id, user_query) for each DPO row."""
    rows: list[tuple[str, str]] = []
    with open(path) as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            sid = str(ex.get("id") or ex.get("source_id") or f"dpo#{i}")
            for msg in ex.get("prompt_messages", []):
                if msg.get("role") == "user":
                    content = msg.get("content", "") or ""
                    if content.strip():
                        rows.append((sid, content))
                    break
    return rows


# ---------------------------------------------------------------------------
# Overlap checks
# ---------------------------------------------------------------------------

def build_train_index(
    train_rows: list[tuple[str, str]],
) -> tuple[MinHashLSH, dict[str, MinHash], dict[str, str], dict[str, list[str]]]:
    """Index training queries into an LSH; also return exact-normalized map."""
    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    key_to_mh: dict[str, MinHash] = {}
    key_to_sid: dict[str, str] = {}
    exact_index: dict[str, list[str]] = {}  # normalized text -> [source_ids]

    for i, (sid, query) in enumerate(train_rows):
        key = f"t{i}"
        mh = make_minhash(query)
        lsh.insert(key, mh)
        key_to_mh[key] = mh
        key_to_sid[key] = sid
        exact_index.setdefault(normalize_exact(query), []).append(sid)
    return lsh, key_to_mh, key_to_sid, exact_index


def check_overlap(
    bfcl_queries: list[tuple[str, str, str]],
    lsh: MinHashLSH,
    key_to_mh: dict[str, MinHash],
    key_to_sid: dict[str, str],
    exact_index: dict[str, list[str]],
) -> tuple[list[dict], list[dict]]:
    """Return (near_dup_hits, exact_hits)."""
    near_hits: list[dict] = []
    exact_hits: list[dict] = []
    for bfcl_id, tag, text in bfcl_queries:
        norm = normalize_exact(text)
        for sid in exact_index.get(norm, []):
            exact_hits.append(
                {"bfcl_id": bfcl_id, "turn": tag, "train_source_id": sid, "text": text[:200]}
            )
        mh = make_minhash(text)
        for key in lsh.query(mh):
            sim = mh.jaccard(key_to_mh[key])
            if sim >= JACCARD_THRESHOLD:
                near_hits.append(
                    {
                        "bfcl_id": bfcl_id,
                        "turn": tag,
                        "train_source_id": key_to_sid[key],
                        "similarity": round(float(sim), 3),
                        "text": text[:200],
                    }
                )
    return near_hits, exact_hits


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    bfcl_files: list[Path],
    bfcl_query_count: int,
    per_file_counts: dict[str, int],
    sft_row_count: int,
    dpo_row_count: int,
    sft_near: list[dict],
    sft_exact: list[dict],
    dpo_near: list[dict],
    dpo_exact: list[dict],
) -> None:
    def _unique_bfcl(hits: list[dict]) -> int:
        return len({h["bfcl_id"] for h in hits})

    lines: list[str] = []
    lines.append("# BFCL v3 leakage report\n")
    lines.append(
        "Automated check that our training data does not overlap the BFCL v3 "
        "eval set. Generated by `eval/bfcl_leakage_check.py`.\n"
    )
    lines.append("## Method\n")
    lines.append(
        "- MinHash+LSH with 128 permutations, word 5-grams, Jaccard threshold "
        f"{JACCARD_THRESHOLD} — identical parameters to `data/dedupe.py` so "
        "results are directly comparable to Week 3 internal dedup.\n"
        "- Exact normalized match (whitespace-collapsed, lowercased) as a "
        "stricter smoking-gun signal.\n"
        "- Applied to user-query text only (the input distribution our model "
        "sees) — matches the dedup rationale in `data/dedupe.py`.\n"
    )
    lines.append("## Sources checked\n")
    lines.append(f"- BFCL files: {len(bfcl_files)}")
    for p in bfcl_files:
        lines.append(f"  - `{p.name}` — {per_file_counts.get(p.name, 0)} extracted user turns")
    lines.append(f"- BFCL user-query turns extracted (total): **{bfcl_query_count}**")
    lines.append(f"- SFT training rows checked: **{sft_row_count}** (from `{SFT_PATH.name}`)")
    lines.append(f"- DPO training rows checked: **{dpo_row_count}** (from `{DPO_PATH.name}`)\n")

    lines.append("## Headline overlap counts\n")
    lines.append("| Training set | Exact matches | Near-dup (Jaccard ≥ 0.7) | Unique BFCL tasks hit |")
    lines.append("|--------------|--------------:|-------------------------:|----------------------:|")
    lines.append(
        f"| SFT (`sft_dedup.jsonl`) | {len(sft_exact)} | {len(sft_near)} | "
        f"{_unique_bfcl(sft_exact + sft_near)} |"
    )
    lines.append(
        f"| DPO (`preferences_dpo.jsonl`) | {len(dpo_exact)} | {len(dpo_near)} | "
        f"{_unique_bfcl(dpo_exact + dpo_near)} |\n"
    )

    def _dump_hits(name: str, hits: list[dict], kind: str) -> None:
        lines.append(f"### {name} — {kind} ({len(hits)})\n")
        if not hits:
            lines.append("_No hits._\n")
            return
        for h in hits[:200]:  # cap per section to keep report readable
            sim = h.get("similarity", "1.000 (exact)")
            lines.append(
                f"- bfcl_id=`{h['bfcl_id']}` turn=`{h['turn']}` "
                f"train_source_id=`{h['train_source_id']}` similarity=`{sim}`"
            )
            lines.append(f"  - text: `{h['text']!r}`")
        if len(hits) > 200:
            lines.append(f"\n_(+{len(hits) - 200} more hits omitted)_\n")

    lines.append("## Matched pairs\n")
    _dump_hits("SFT", sft_exact, "exact matches")
    _dump_hits("SFT", sft_near, "near-duplicates")
    _dump_hits("DPO", dpo_exact, "exact matches")
    _dump_hits("DPO", dpo_near, "near-duplicates")

    lines.append("## Interpretation\n")
    lines.append(
        "- **Any exact hit** is a hard leakage: remove the offending training "
        "row(s) and re-run SFT/DPO before publishing scores.\n"
        "- **Near-dup hits** should each be inspected — some are genuine "
        "leakage, others may be common tool-calling phrasings that "
        "coincidentally overlap.\n"
        "- **Zero hits** does not fully rule out leakage (e.g., semantic "
        "paraphrases below 0.7 Jaccard), but combined with the Week 3 dedup "
        "history it is a strong signal.\n"
    )

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="Force re-download of BFCL files.")
    ap.add_argument("--offline", action="store_true", help="Skip download; use cache only.")
    args = ap.parse_args()

    if not SFT_PATH.exists():
        raise SystemExit(f"Missing {SFT_PATH}")
    if not DPO_PATH.exists():
        raise SystemExit(f"Missing {DPO_PATH}")

    bfcl_files = download_bfcl(refresh=args.refresh, offline=args.offline)
    if not bfcl_files:
        raise SystemExit("No BFCL files downloaded or cached; cannot proceed.")

    # Extract BFCL queries
    all_bfcl: list[tuple[str, str, str]] = []
    per_file_counts: dict[str, int] = {}
    for p in bfcl_files:
        rows = extract_bfcl_queries(p)
        per_file_counts[p.name] = len(rows)
        all_bfcl.extend(rows)
    print(f"Extracted {len(all_bfcl)} BFCL user turns across {len(bfcl_files)} files")

    # Extract training queries + build indexes
    sft_rows = extract_sft_queries(SFT_PATH)
    dpo_rows = extract_dpo_queries(DPO_PATH)
    print(f"Indexing {len(sft_rows)} SFT queries...")
    sft_idx = build_train_index(sft_rows)
    print(f"Indexing {len(dpo_rows)} DPO queries...")
    dpo_idx = build_train_index(dpo_rows)

    # Run checks
    print("Checking SFT overlap...")
    sft_near, sft_exact = check_overlap(all_bfcl, *sft_idx)
    print(f"  SFT exact: {len(sft_exact)}  near-dup: {len(sft_near)}")
    print("Checking DPO overlap...")
    dpo_near, dpo_exact = check_overlap(all_bfcl, *dpo_idx)
    print(f"  DPO exact: {len(dpo_exact)}  near-dup: {len(dpo_near)}")

    write_report(
        bfcl_files=bfcl_files,
        bfcl_query_count=len(all_bfcl),
        per_file_counts=per_file_counts,
        sft_row_count=len(sft_rows),
        dpo_row_count=len(dpo_rows),
        sft_near=sft_near,
        sft_exact=sft_exact,
        dpo_near=dpo_near,
        dpo_exact=dpo_exact,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
