"""BFCL v4 single-call eval: SFT baseline vs DPO checkpoint sweep.

Supports `simple_python` and `multiple`, whose answer keys both contain one
expected call per item. The simplified scorer uses exact function-name + strict
per-arg value-in-accepted-list match + no-extra-args. Not identical to BFCL's
official AST-based scorer; sufficient for tracking relative SFT-vs-DPO deltas
on these categories. Full BFCL leaderboard submission is out of scope for
llama-tools v1.

For the SFT baseline plus each DPO checkpoint, this script runs greedy
generation, extracts the tool call as JSON,
and scores name + arguments against the accepted-values ground truth. Writes:

- `eval/out/bfcl_simple/generations.jsonl` — one row per (id, candidate)
- `eval/out/bfcl_simple/report.md` — summary table

Usage:
    python eval/bfcl_simple.py
    python eval/bfcl_simple.py --category multiple --sft-only
    python eval/bfcl_simple.py --num-prompts 10
    python eval/bfcl_simple.py --checkpoints 50 100 150 --checkpoint-root outputs/dpo-v2-full
    python eval/bfcl_simple.py --sft-only
"""

import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from bfcl_category_config import SUPPORTED_CATEGORIES, resolve_category_paths

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
SFT_ADAPTER_DIR = Path("./outputs/sft-full")
CHECKPOINT_ROOT = Path("./outputs/dpo-v2-full")
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _git_head() -> str:
    """Exact code revision this run executed, or a loud marker if unknown."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "UNKNOWN"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_run_manifest(out_dir: Path, args, candidates: list, gen_path: Path,
                       category_paths, n_prompts: int) -> Path:
    """Record what produced this evidence, so a number can be traced to a run.

    Written next to the generations rather than derived later: a reviewer
    asking "which code, which weights, which inputs produced 369/400?" must be
    able to answer it from the artifact alone.
    """
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "code_revision": _git_head(),
        "cli": " ".join(sys.argv),
        "category": args.category,
        "candidates": candidates,
        "n_prompts": n_prompts,
        "expected_rows": n_prompts * len(candidates),
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "sft_adapter": args.sft_adapter,
        "sft_adapter_subfolder": args.sft_adapter_subfolder,
        "sft_adapter_revision": args.sft_adapter_revision,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "inputs": {
            "questions": {
                "path": str(category_paths.questions),
                "sha256": _sha256_file(category_paths.questions),
            },
            "answer_key": {
                "path": str(category_paths.answer_key),
                "sha256": _sha256_file(category_paths.answer_key),
            },
        },
        "outputs": {
            "generations": {"path": str(gen_path), "sha256": _sha256_file(gen_path)},
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": getattr(torch, "__version__", "unknown"),
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def build_candidate_names(include_base: bool, sft_only: bool, checkpoints) -> list:
    """Ordered candidate names for a sweep.

    "base" is not an adapter: it denotes the unmodified base model, generated
    with adapters disabled. It must therefore never be passed to set_adapter().
    "sft" is always scored — it is the study-1 shipped baseline every other
    candidate is compared against.
    """
    names = ["base"] if include_base else []
    names.append("sft")
    if not sft_only:
        names.extend(f"dpo-{step}" for step in (checkpoints or []))
    return names


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
    parser.add_argument(
        "--category", choices=SUPPORTED_CATEGORIES, default="simple_python"
    )
    parser.add_argument("--num-prompts", type=int, default=None,
                        help="Limit for smoke tests; default = all category items")
    parser.add_argument("--checkpoints", type=int, nargs="+",
                        default=[50, 100, 150])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_MODEL_REVISION)
    parser.add_argument("--sft-adapter", default=str(SFT_ADAPTER_DIR))
    parser.add_argument("--sft-adapter-subfolder", default=None)
    parser.add_argument("--sft-adapter-revision", default=None)
    parser.add_argument("--include-base", action="store_true",
                        help="Also score the unmodified base model (adapters disabled). "
                             "Required for base-vs-SFT lift numbers.")
    parser.add_argument("--sft-only", action="store_true",
                        help="Skip DPO checkpoints; baseline only")
    parser.add_argument("--overwrite", action="store_true",
                        help="Discard existing evidence in --out-dir. Refuses by default.")
    args = parser.parse_args()

    load_dotenv()
    category_paths = resolve_category_paths(REPO_ROOT, args.category)
    if args.out_dir is None:
        args.out_dir = category_paths.default_output

    # Refuse to write into a directory that already holds results. A paid run
    # silently overwriting a previous paid run's evidence is unrecoverable:
    # the generations are the only record of what the model actually emitted.
    gen_path = args.out_dir / "generations.jsonl"
    if gen_path.exists() and gen_path.stat().st_size > 0 and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite existing evidence at {gen_path}\n"
            f"use a fresh --out-dir, or pass --overwrite if you truly mean to discard it"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading BFCL prompts: {category_paths.questions}")
    prompts_raw = load_jsonl(category_paths.questions)
    gt_raw = load_jsonl(category_paths.answer_key)
    gt_by_id = {r["id"]: r["ground_truth"][0] for r in gt_raw}
    prompt_ids = {row["id"] for row in prompts_raw}
    if prompt_ids != set(gt_by_id):
        missing_keys = sorted(prompt_ids - set(gt_by_id))
        extra_keys = sorted(set(gt_by_id) - prompt_ids)
        raise ValueError(
            f"question/answer id mismatch: missing={missing_keys[:5]} "
            f"extra={extra_keys[:5]}"
        )

    if args.num_prompts is not None:
        prompts_raw = prompts_raw[: args.num_prompts]
    print(f"Evaluating on {len(prompts_raw)} prompts")

    print(f"Loading tokenizer: {args.base_model} revision={args.base_revision}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, revision=args.base_revision
    )
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

    print(f"Loading base model: {args.base_model} revision={args.base_revision}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print(
        f"Attaching SFT adapter: {args.sft_adapter} "
        f"subfolder={args.sft_adapter_subfolder} "
        f"revision={args.sft_adapter_revision}"
    )
    adapter_kwargs = {}
    if args.sft_adapter_subfolder:
        adapter_kwargs["subfolder"] = args.sft_adapter_subfolder
    if args.sft_adapter_revision:
        adapter_kwargs["revision"] = args.sft_adapter_revision
    model = PeftModel.from_pretrained(
        base, args.sft_adapter, adapter_name="sft", **adapter_kwargs
    )

    # "base" is the unmodified base model: same weights, adapters disabled at
    # generation time. It is not an adapter, so it is never passed to set_adapter().
    candidates = build_candidate_names(
        include_base=args.include_base,
        sft_only=args.sft_only,
        checkpoints=args.checkpoints,
    )
    if not args.sft_only:
        for step in args.checkpoints:
            ckpt_dir = args.checkpoint_root / f"checkpoint-{step}"
            if not ckpt_dir.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_dir}")
            name = f"dpo-{step}"
            print(f"Loading adapter {name} from {ckpt_dir}")
            model.load_adapter(str(ckpt_dir), adapter_name=name)

    model.eval()

    # Evidence is written incrementally, not buffered until the end. A paid
    # run that dies at 90% must leave 90% of its generations on disk; the old
    # behaviour buffered everything in memory and wrote once per category, so
    # a preempted pod lost the entire spend.
    rows = []
    gen_file = gen_path.open("w", encoding="utf-8")

    def checkpoint(row: dict) -> None:
        gen_file.write(json.dumps(row) + "\n")
        gen_file.flush()
        os.fsync(gen_file.fileno())

    for cand in candidates:
        if cand == "base":
            adapter_ctx = model.disable_adapter()
        else:
            model.set_adapter(cand)
            adapter_ctx = contextlib.nullcontext()
        print(f"\n=== Generating with {cand} ===")
        with adapter_ctx:
            for i, ex in enumerate(built_prompts):
                gen = generate(model, tokenizer, ex["prompt"], args.max_new_tokens)
                parsed = extract_json(gen)
                is_obj = isinstance(parsed, dict)
                gt_entry = gt_by_id[ex["id"]]
                name_ok, args_ok, overall_ok, reason = score(parsed, gt_entry)
                rows.append(
                    {
                        "id": ex["id"],
                        "category": args.category,
                        "model_name": cand,
                        "output": gen,
                        "parsed_name": parsed.get("name") if is_obj else None,
                        "parsed_args": parsed.get("arguments") if is_obj else None,
                        "name_ok": name_ok,
                        "args_ok": args_ok,
                        "overall_ok": overall_ok,
                        "failure_reason": reason,
                        "json_valid": parsed is not None,
                    }
                )
                checkpoint(rows[-1])
                if (i + 1) % 25 == 0 or (i + 1) == len(built_prompts):
                    print(
                        f"  [{i + 1}/{len(built_prompts)}] "
                        f"overall={rows[-1]['overall_ok']} reason={reason or 'ok'}"
                    )

    gen_file.close()

    # Completeness assertion: every prompt must have been scored by every
    # candidate. A short file means the run was cut off, and a silently short
    # evidence file would produce a wrong accuracy denominator downstream.
    expected = len(built_prompts) * len(candidates)
    if len(rows) != expected:
        raise SystemExit(
            f"incomplete run: {len(rows)} rows written, expected {expected} "
            f"({len(built_prompts)} prompts x {len(candidates)} candidates). "
            f"Partial evidence is preserved at {gen_path}."
        )
    seen_pairs = {(r["id"], r["model_name"]) for r in rows}
    if len(seen_pairs) != expected:
        raise SystemExit(
            f"duplicate or missing (id, candidate) pairs: {len(seen_pairs)} distinct "
            f"of {expected} expected. Evidence preserved at {gen_path}."
        )
    print(f"\nWrote {gen_path} ({len(rows)} rows, verified complete)")

    manifest_path = write_run_manifest(
        args.out_dir, args, candidates, gen_path, category_paths, len(built_prompts)
    )
    print(f"Wrote {manifest_path}")

    n = len(built_prompts)
    lines = [f"# BFCL v4 {args.category} — SFT vs DPO sweep", ""]
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

    print(f"\n=== BFCL {args.category} summary ===")
    for line in lines[2 : 4 + len(candidates)]:
        print(line)


if __name__ == "__main__":
    main()
