"""MMLU 5-shot regression eval: base vs SFT baseline vs DPO checkpoint sweep.

Self-contained minimal MMLU runner. No lm-evaluation-harness dependency.
Loads `cais/mmlu` from Hugging Face, builds standard 5-shot prompts (few-shot
examples sampled from the `dev` split of the same subject as the test item),
and scores each item by comparing next-token log-probs of the four choice
letters (" A", " B", " C", " D") and taking argmax. This matches how
lm-eval-harness scores MMLU. Writes:

- `eval/out/mmlu_regression/predictions.jsonl` — one row per (model, item)
- `eval/out/mmlu_regression/report.md` — summary table (macro-avg accuracy
  across all items, plus delta vs base)

Usage:
    python eval/mmlu_regression.py
    python eval/mmlu_regression.py --num-items 500
    python eval/mmlu_regression.py --checkpoints 50 150
    python eval/mmlu_regression.py --base-only
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SFT_ADAPTER_DIR = Path("./outputs/sft-full")
DEFAULT_CHECKPOINT_ROOT = Path("./outputs/dpo-v2-full")

OUT_DIR = Path("./eval/out/mmlu_regression")

MMLU_DATASET = "cais/mmlu"
NUM_FEW_SHOT = 5
CHOICES = ["A", "B", "C", "D"]
FEW_SHOT_SEED = 1234


def format_subject(subject: str) -> str:
    """`abstract_algebra` -> `abstract algebra`."""
    return subject.replace("_", " ")


def format_example(question: str, choices, answer_idx) -> str:
    """Render one MMLU item. If answer_idx is None, leaves `Answer:` open."""
    lines = [question]
    for letter, choice in zip(CHOICES, choices):
        lines.append(f"{letter}. {choice}")
    if answer_idx is None:
        lines.append("Answer:")
    else:
        lines.append(f"Answer: {CHOICES[answer_idx]}")
    return "\n".join(lines)


def build_prompt(test_item, few_shot_examples) -> str:
    subject_str = format_subject(test_item["subject"])
    header = (
        f"The following are multiple choice questions (with answers) "
        f"about {subject_str}.\n"
    )
    blocks = [header]
    for fs in few_shot_examples:
        blocks.append(
            format_example(fs["question"], fs["choices"], fs["answer"]) + "\n"
        )
    blocks.append(format_example(test_item["question"], test_item["choices"], None))
    return "\n".join(blocks)


def build_few_shot_pool(dev_split):
    """Group dev-split items by subject for few-shot sampling.

    Note: MMLU dev split ships ~5 examples per subject by design, but a
    handful of subjects fall short. We fall back to sampling with
    replacement so the prompt always has NUM_FEW_SHOT shots; this is a
    rare edge case and matches what lm-eval-harness does in practice.
    """
    pool = defaultdict(list)
    for item in dev_split:
        pool[item["subject"]].append(item)
    return pool


def sample_few_shot(pool, subject: str, rng: random.Random):
    candidates = pool.get(subject, [])
    if len(candidates) >= NUM_FEW_SHOT:
        return rng.sample(candidates, NUM_FEW_SHOT)
    if not candidates:
        # extremely defensive — every MMLU subject has dev entries in practice
        return []
    return [rng.choice(candidates) for _ in range(NUM_FEW_SHOT)]


def resolve_choice_token_ids(tokenizer):
    """Return the token ids for ' A', ' B', ' C', ' D'.

    Verified: for the Llama-3.1 tokenizer (tiktoken-based BPE), each of
    " A", " B", " C", " D" encodes to a single token — so taking the last
    token id after `add_special_tokens=False` is safe. Asserted at runtime.
    """
    ids = []
    for letter in CHOICES:
        toks = tokenizer.encode(" " + letter, add_special_tokens=False)
        assert len(toks) == 1, (
            f"Expected single token for ' {letter}', got {toks}. "
            "Scoring assumes a single-token encoding."
        )
        ids.append(toks[0])
    return ids


@torch.no_grad()
def score_mmlu_item(model, tokenizer, prompt_text: str, choice_token_ids) -> int:
    """Return predicted answer index 0-3 via argmax over choice-letter logits."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1]
    choice_logits = logits[choice_token_ids]
    return int(choice_logits.argmax().item())


def _run_candidate(model, tokenizer, built, choice_token_ids, cand, rows, n):
    print(f"\n=== Scoring with {cand} ===")
    correct_running = 0
    for i, ex in enumerate(built):
        pred = score_mmlu_item(model, tokenizer, ex["prompt"], choice_token_ids)
        is_correct = pred == ex["answer"]
        correct_running += int(is_correct)
        rows.append(
            {
                "idx": ex["idx"],
                "model_name": cand,
                "subject": ex["subject"],
                "answer": ex["answer"],
                "predicted": pred,
                "correct": is_correct,
            }
        )
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(
                f"  [{i + 1}/{n}] running_acc={correct_running / (i + 1):.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-items", type=int, default=None,
                        help="Random subsample size; default = full test set")
    parser.add_argument("--checkpoints", type=int, nargs="+",
                        default=[50, 100, 150])
    parser.add_argument("--checkpoint-root", type=Path,
                        default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--base-only", action="store_true",
                        help="Skip SFT + DPO; base baseline only")
    parser.add_argument("--seed", type=int, default=FEW_SHOT_SEED)
    args = parser.parse_args()

    load_dotenv()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MMLU test split: {MMLU_DATASET}")
    test_ds = load_dataset(MMLU_DATASET, "all", split="test")
    dev_ds = load_dataset(MMLU_DATASET, "all", split="dev")

    test_items = list(test_ds)
    if args.num_items is not None and args.num_items < len(test_items):
        subsample_rng = random.Random(args.seed)
        test_items = subsample_rng.sample(test_items, args.num_items)
    print(f"Evaluating on {len(test_items)} items "
          f"(dev pool size {len(dev_ds)})")

    few_shot_pool = build_few_shot_pool(dev_ds)

    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    choice_token_ids = resolve_choice_token_ids(tokenizer)

    # Pre-build prompts once with a fixed seed so every model sees the same
    # few-shot examples per test item.
    prompt_rng = random.Random(args.seed)
    built = []
    for idx, item in enumerate(test_items):
        fs = sample_few_shot(few_shot_pool, item["subject"], prompt_rng)
        built.append(
            {
                "idx": idx,
                "subject": item["subject"],
                "answer": int(item["answer"]),
                "prompt": build_prompt(item, fs),
            }
        )

    print(f"Loading base model: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto"
    )
    base.eval()

    candidates = ["base"]
    model = base  # base-only path

    if not args.base_only:
        print(f"Attaching SFT adapter: {SFT_ADAPTER_DIR}")
        model = PeftModel.from_pretrained(
            base, str(SFT_ADAPTER_DIR), adapter_name="sft"
        )
        candidates.append("sft")
        for step in args.checkpoints:
            ckpt_dir = args.checkpoint_root / f"checkpoint-{step}"
            if not ckpt_dir.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_dir}")
            name = f"dpo-{step}"
            print(f"Loading adapter {name} from {ckpt_dir}")
            model.load_adapter(str(ckpt_dir), adapter_name=name)
            candidates.append(name)
        model.eval()

    rows = []
    n = len(built)
    for cand in candidates:
        if cand == "base":
            if isinstance(model, PeftModel):
                # Disable all adapters for a clean base pass.
                with model.disable_adapter():
                    _run_candidate(model, tokenizer, built, choice_token_ids,
                                   cand, rows, n)
                continue
            active_model = base
        else:
            model.set_adapter(cand)
            active_model = model
        _run_candidate(active_model, tokenizer, built, choice_token_ids,
                       cand, rows, n)

    pred_path = args.out_dir / "predictions.jsonl"
    with pred_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {pred_path}")

    # Aggregate: overall accuracy (macro-avg across items) per candidate.
    per_cand = {}
    for cand in candidates:
        sub = [r for r in rows if r["model_name"] == cand]
        correct = sum(r["correct"] for r in sub)
        per_cand[cand] = (correct, len(sub))

    base_acc = None
    if "base" in per_cand and per_cand["base"][1]:
        base_acc = per_cand["base"][0] / per_cand["base"][1]

    lines = [
        "# MMLU 5-shot — base vs SFT vs DPO sweep",
        "",
        "| candidate | mmlu (5-shot) | delta vs base |",
        "|---|---|---|",
    ]
    for cand in candidates:
        correct, total = per_cand[cand]
        acc = correct / total if total else 0.0
        if cand == "base" or base_acc is None:
            delta = "—"
        else:
            delta = f"{(acc - base_acc):+.3f}"
        lines.append(f"| {cand} | {acc:.3f} ({correct}/{total}) | {delta} |")
    lines.append("")

    report_path = args.out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path}")

    print("\n=== MMLU 5-shot summary ===")
    for line in lines[2 : 4 + len(candidates)]:
        print(line)


if __name__ == "__main__":
    main()
