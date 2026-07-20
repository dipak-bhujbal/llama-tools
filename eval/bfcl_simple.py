"""BFCL v4 simple_python eval: SFT baseline vs DPO checkpoint sweep.

Simplified BFCL v4 simple_python scorer: exact function-name match + strict
per-arg value-in-accepted-list match + no-extra-args. Not identical to BFCL's
official AST-based scorer; sufficient for tracking relative SFT-vs-DPO deltas
on this category. Full BFCL leaderboard submission out of scope for
llama-tools v1.

For the SFT baseline plus each DPO checkpoint, this script runs greedy
generation on 399 BFCL simple_python prompts, extracts the tool call as JSON,
and scores name + arguments against the accepted-values ground truth. Writes:

- `eval/out/bfcl_simple/generations.jsonl` — one row per (id, candidate)
- `eval/out/bfcl_simple/report.md` — summary table

Usage:
    python eval/bfcl_simple.py
    python eval/bfcl_simple.py --num-prompts 10
    python eval/bfcl_simple.py --checkpoints 50 100 150 --checkpoint-root outputs/dpo-v2-full
    python eval/bfcl_simple.py --sft-only
"""

import argparse
import json
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SFT_ADAPTER_DIR = Path("./outputs/sft-full")
CHECKPOINT_ROOT = Path("./outputs/dpo-v2-full")

PROMPTS_PATH = Path("./eval/bfcl_data/BFCL_v4_simple_python.json")
GT_PATH = Path("./eval/bfcl_data/possible_answer/BFCL_v4_simple_python.json")

OUT_DIR = Path("./eval/out/bfcl_simple")


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(tokenizer, question, functions) -> str:
    tool_json = json.dumps(functions, indent=2)
    system = (
        "You have access to the following tools:\n"
        f"{tool_json}\n"
        'Respond with a single JSON object of the form '
        '{"name": <function_name>, "arguments": {<kwargs>}}'
    )
    messages = [{"role": "system", "content": system}]
    # question is [[{role, content}, ...]] — single turn for simple_python
    messages.extend(question[0])
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_json(text: str):
    """Find first `{` to last `}` and try json.loads. Return dict or None."""
    text = text.strip()
    if "{" not in text or "}" not in text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def values_equal(parsed_val, accepted_val) -> bool:
    """Light coercion equality: numeric compare if both numbers, else deep =="""
    if isinstance(parsed_val, bool) or isinstance(accepted_val, bool):
        return parsed_val == accepted_val
    if isinstance(parsed_val, (int, float)) and isinstance(accepted_val, (int, float)):
        return float(parsed_val) == float(accepted_val)
    return parsed_val == accepted_val


def score(parsed, gt_entry):
    """Return (name_ok, args_ok, overall_ok, failure_reason).

    gt_entry: {"function_name": {"arg1": [accepted, ...], "arg2": [...]}}
    """
    if parsed is None:
        return False, False, False, "json_unparseable"
    if "name" not in parsed or "arguments" not in parsed:
        return False, False, False, "missing_name_or_arguments"
    parsed_name = parsed["name"]
    parsed_args = parsed["arguments"]
    if not isinstance(parsed_args, dict):
        return False, False, False, "arguments_not_dict"

    gt_name = next(iter(gt_entry.keys()))
    gt_args = gt_entry[gt_name]
    name_ok = parsed_name == gt_name

    # no-extra-args: every parsed key must be in gt
    for k in parsed_args:
        if k not in gt_args:
            return name_ok, False, False, f"extra_arg:{k}"

    # each required gt arg must match; optional means "" in accepted list
    args_ok = True
    fail_reason = ""
    for arg_name, accepted in gt_args.items():
        optional = "" in accepted
        if arg_name not in parsed_args:
            if optional:
                continue
            args_ok = False
            fail_reason = f"missing_arg:{arg_name}"
            break
        parsed_val = parsed_args[arg_name]
        if not any(values_equal(parsed_val, av) for av in accepted):
            args_ok = False
            fail_reason = f"bad_value:{arg_name}"
            break

    overall_ok = name_ok and args_ok
    if overall_ok:
        return True, True, True, ""
    if not name_ok and not fail_reason:
        fail_reason = "bad_name"
    return name_ok, args_ok, overall_ok, fail_reason


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=None,
                        help="Limit for smoke tests; default = all 399")
    parser.add_argument("--checkpoints", type=int, nargs="+",
                        default=[50, 100, 150])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--sft-only", action="store_true",
                        help="Skip DPO checkpoints; baseline only")
    args = parser.parse_args()

    load_dotenv()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading BFCL prompts: {PROMPTS_PATH}")
    prompts_raw = load_jsonl(PROMPTS_PATH)
    gt_raw = load_jsonl(GT_PATH)
    gt_by_id = {r["id"]: r["ground_truth"][0] for r in gt_raw}

    if args.num_prompts is not None:
        prompts_raw = prompts_raw[: args.num_prompts]
    print(f"Evaluating on {len(prompts_raw)} prompts")

    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-build prompt strings once
    built_prompts = []
    for ex in prompts_raw:
        built_prompts.append(
            {
                "id": ex["id"],
                "prompt": build_prompt(tokenizer, ex["question"], ex["function"]),
            }
        )

    print(f"Loading base model: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto"
    )

    print(f"Attaching SFT adapter: {SFT_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base, str(SFT_ADAPTER_DIR), adapter_name="sft")

    candidates = ["sft"]
    if not args.sft_only:
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
    for cand in candidates:
        model.set_adapter(cand)
        print(f"\n=== Generating with {cand} ===")
        for i, ex in enumerate(built_prompts):
            gen = generate(model, tokenizer, ex["prompt"], args.max_new_tokens)
            parsed = extract_json(gen)
            gt_entry = gt_by_id[ex["id"]]
            name_ok, args_ok, overall_ok, reason = score(parsed, gt_entry)
            rows.append(
                {
                    "id": ex["id"],
                    "model_name": cand,
                    "output": gen,
                    "parsed_name": parsed.get("name") if isinstance(parsed, dict) else None,
                    "parsed_args": parsed.get("arguments") if isinstance(parsed, dict) else None,
                    "name_ok": name_ok,
                    "args_ok": args_ok,
                    "overall_ok": overall_ok,
                    "failure_reason": reason,
                    "json_valid": parsed is not None,
                }
            )
            if (i + 1) % 25 == 0 or (i + 1) == len(built_prompts):
                print(
                    f"  [{i + 1}/{len(built_prompts)}] "
                    f"overall={rows[-1]['overall_ok']} reason={reason or 'ok'}"
                )

    gen_path = args.out_dir / "generations.jsonl"
    with gen_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {gen_path}")

    n = len(built_prompts)
    lines = ["# BFCL v4 simple_python — SFT vs DPO sweep", ""]
    lines += [
        "| candidate | overall | name_ok | args_ok | json_valid |",
        "|---|---|---|---|---|",
    ]
    for cand in candidates:
        sub = [r for r in rows if r["model_name"] == cand]
        overall = sum(r["overall_ok"] for r in sub)
        name_ok = sum(r["name_ok"] for r in sub)
        args_ok = sum(r["args_ok"] for r in sub)
        jv = sum(r["json_valid"] for r in sub)
        lines.append(
            f"| {cand} | {overall}/{n} | {name_ok}/{n} | {args_ok}/{n} | {jv}/{n} |"
        )
    lines.append("")

    report_path = args.out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path}")

    print("\n=== BFCL simple_python summary ===")
    for line in lines[2 : 4 + len(candidates)]:
        print(line)


if __name__ == "__main__":
    main()
