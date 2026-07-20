"""DPO v2 Stage 1: sample the SFT policy's own failures as candidate rejecteds.

Implements ADR-007. For each unique training-split prompt from the v1
preference set, draw K temperature samples from the SFT policy and score them
against the gold `chosen` with semantic (canonical-JSON) comparison. Every
genuine failure is written out as a candidate rejected, tagged with its
failure category:

- invalid_json  — output does not parse as a tool-call structure
- wrong_tool    — parsed, but the set of tool names differs from gold
- wrong_args    — right tools, wrong/missing/extra argument values

Samples canonically equal to gold are successes and are dropped. The
300-prompt trainer holdout (same seed/split as train/dpo_full.py and
eval/dpo_sweep.py) is NEVER sampled, so the final judging sweep stays clean.

Prereqs on the pod:
- SFT adapter at `outputs/sft-full/`
- `data/processed/preferences_dpo.jsonl` present
- `pip install -e ".[train]"`, `hf auth login --token $HF_TOKEN`

Usage:
    python data/sample_failures.py
    python data/sample_failures.py --max-prompts 2000 --samples-per-prompt 4
    # resume-friendly: skips prompts already present in the output file
    python data/sample_failures.py --resume

Output: data/processed/failure_candidates.jsonl
    {prompt_messages, chosen, rejected, failure_type, source, source_id}
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SFT_ADAPTER_DIR = Path("./outputs/sft-full")

# Must match train/dpo_full.py exactly (holdout exclusion depends on it).
DATASET_PATH = "data/processed/preferences_dpo.jsonl"
EVAL_SPLIT = 300
SPLIT_SEED = 42

OUT_PATH = Path("data/processed/failure_candidates.jsonl")

TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 512
BATCH_SIZE = 16
SAMPLE_SEED = 42


def _sort_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_keys(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_sort_keys(x) for x in obj]
    return obj


def _canonicalize(text: str) -> Any:
    """Parse tool-call text to a canonical structure; None if unparseable.

    Mirrors data/validate_preferences.py, plus a brace-slice fallback for
    generations that prepend prose before the JSON (same leniency as
    eval/dpo_sweep.py's try_json)."""
    text = text.strip()
    candidates = [text]
    for opener in ("[", "{"):
        idx = text.find(opener)
        if idx > 0:
            candidates.append(text[idx:])
    for cand in candidates:
        try:
            return _sort_keys(json.loads(cand))
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _tool_names(canon: Any) -> list[str] | None:
    """Extract sorted tool-name multiset from a canonical parsed structure."""
    calls = canon if isinstance(canon, list) else [canon]
    names = []
    for call in calls:
        if not isinstance(call, dict) or "name" not in call:
            return None
        names.append(str(call["name"]))
    return sorted(names)


def classify(generation: str, chosen: str) -> str | None:
    """Return failure category, or None if the generation is a success."""
    g_canon = _canonicalize(generation)
    c_canon = _canonicalize(chosen)
    if g_canon is None:
        return "invalid_json"
    if c_canon is not None and g_canon == c_canon:
        return None  # success
    g_names, c_names = _tool_names(g_canon), _tool_names(c_canon)
    if g_names is None or (c_names is not None and g_names != c_names):
        return "wrong_tool"
    return "wrong_args"


def load_train_prompts(tokenizer, max_prompts: int):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=DATASET_PATH, split="train")

    def to_row(row):
        prompt = tokenizer.apply_chat_template(
            row["prompt_messages"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt}

    ds = ds.map(to_row)
    # Identical split call to train/dpo_full.py — "train" side only, so the
    # 300-prompt holdout is never sampled.
    train = ds.train_test_split(test_size=EVAL_SPLIT, seed=SPLIT_SEED)["train"]

    # v1 reused source examples across pairs; sample each unique prompt once.
    seen: set[str] = set()
    rows = []
    for ex in train:
        key = ex["source_id"]
        if key in seen:
            continue
        seen.add(key)
        rows.append(ex)
        if len(rows) >= max_prompts:
            break
    return rows


@torch.no_grad()
def sample_batch(model, tokenizer, prompts: list[str], k: int) -> list[list[str]]:
    """Return k sampled completions per prompt."""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, padding_side="left"
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        num_return_sequences=k,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    texts = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    # generate() interleaves: k sequences per prompt, prompt-major order.
    return [texts[i * k : (i + 1) * k] for i in range(len(prompts))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-prompts", type=int, default=9000)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resume", action="store_true",
                        help="Skip prompts already present in the output file.")
    args = parser.parse_args()

    load_dotenv()
    torch.manual_seed(SAMPLE_SEED)

    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_train_prompts(tokenizer, args.max_prompts)
    print(f"Unique training-split prompts to sample: {len(rows)}")

    done: set[str] = set()
    if args.resume and OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for line in f:
                done.add(json.loads(line).get("_sampled_source_id", ""))
        print(f"Resume: {len(done)} source_ids already sampled")

    print(f"Loading base model: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, str(SFT_ADAPTER_DIR))
    model.eval()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    stats = {"prompts": 0, "success": 0, "invalid_json": 0,
             "wrong_tool": 0, "wrong_args": 0}

    with OUT_PATH.open(mode) as f:
        batch: list[dict] = []

        def flush(batch: list[dict]) -> None:
            prompts = [b["prompt"] for b in batch]
            all_samples = sample_batch(
                model, tokenizer, prompts, args.samples_per_prompt
            )
            for row, samples in zip(batch, all_samples):
                stats["prompts"] += 1
                # A sampled-marker row per prompt lets --resume skip it even
                # when every sample succeeded (no failure rows written).
                f.write(json.dumps({"_sampled_source_id": row["source_id"]}) + "\n")
                seen_rejects: set[str] = set()
                for gen in samples:
                    cat = classify(gen, row["chosen"])
                    if cat is None:
                        stats["success"] += 1
                        continue
                    key = json.dumps(_canonicalize(gen), sort_keys=True) \
                        if _canonicalize(gen) is not None else gen.strip()
                    if key in seen_rejects:
                        continue
                    seen_rejects.add(key)
                    stats[cat] += 1
                    f.write(json.dumps({
                        "prompt_messages": row["prompt_messages"],
                        "chosen": row["chosen"],
                        "rejected": gen.strip(),
                        "failure_type": cat,
                        "source": row.get("source", ""),
                        "source_id": row["source_id"],
                    }) + "\n")
            f.flush()
            done_n = stats["prompts"]
            fails = stats["invalid_json"] + stats["wrong_tool"] + stats["wrong_args"]
            print(f"  [{done_n}/{len(rows)}] failures so far: {fails} "
                  f"(inv={stats['invalid_json']} tool={stats['wrong_tool']} "
                  f"args={stats['wrong_args']}) successes: {stats['success']}")

        for row in rows:
            if row["source_id"] in done:
                continue
            batch.append(row)
            if len(batch) >= args.batch_size:
                flush(batch)
                batch = []
        if batch:
            flush(batch)

    print("\n=== Sampling summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    total_gen = stats["prompts"] * args.samples_per_prompt
    fails = stats["invalid_json"] + stats["wrong_tool"] + stats["wrong_args"]
    if total_gen:
        print(f"  failure rate: {fails}/{total_gen} = {fails / total_gen:.1%}")
    print(f"Wrote {OUT_PATH}")
    print("Next: python data/build_dpo_v2_pairs.py")


if __name__ == "__main__":
    main()
