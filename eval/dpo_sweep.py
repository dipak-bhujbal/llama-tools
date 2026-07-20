"""Week 7 DPO checkpoint sweep: real generations from every vaulted checkpoint.

The Week 6 full DPO run was deliberately stopped at step 400/622 after
convergence, with `eval_rewards/chosen` drifting negative (-0.36 -> -0.42 ->
-0.53 toward the -0.6 kill line). Training metrics alone cannot pick the
winning checkpoint — margins were near-identical from step 200 onward while
chosen-reward degraded. This script settles it with actual generations.

For the SFT baseline plus each DPO checkpoint (100/200/300/400), it generates
greedy completions on the same held-out eval prompts the trainer used
(identical seed + split as train/dpo_full.py), then writes:

- `eval/out/dpo_sweep/generations.jsonl` — one row per (prompt, candidate)
- `eval/out/dpo_sweep/sweep_report.md`  — side-by-side per prompt + summary
  table (exact-match vs chosen, best-effort JSON validity of the output)

The report is the artifact for human winner selection; the metrics are only
a first-pass signal (exact-match is meaningful because chosen completions
are deterministic tool calls).

Prereqs on the pod (see chat runbook):
- SFT adapter at `outputs/sft-full/`
- DPO checkpoints at `outputs/dpo-checkpoints/checkpoint-{100,200,300,400}/`
- `data/processed/preferences_dpo.jsonl` present

Usage:
    python eval/dpo_sweep.py
    python eval/dpo_sweep.py --num-prompts 20 --checkpoints 100 200 300 400
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SFT_ADAPTER_DIR = Path("./outputs/sft-full")
CHECKPOINT_ROOT = Path("./outputs/dpo-checkpoints")

# Must match train/dpo_full.py exactly so we sweep on the trainer's own holdout.
DATASET_PATH = "data/processed/preferences_dpo.jsonl"
EVAL_SPLIT = 300
SPLIT_SEED = 42

OUT_DIR = Path("./eval/out/dpo_sweep")


def load_eval_prompts(tokenizer, num_prompts: int):
    ds = load_dataset("json", data_files=DATASET_PATH, split="train")

    def to_dpo(row):
        prompt = tokenizer.apply_chat_template(
            row["prompt_messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}

    keep = {"prompt", "chosen", "rejected"}
    drop = [c for c in ds.column_names if c not in keep]
    ds = ds.map(to_dpo, remove_columns=drop)
    eval_ds = ds.train_test_split(test_size=EVAL_SPLIT, seed=SPLIT_SEED)["test"]
    return eval_ds.select(range(min(num_prompts, len(eval_ds))))


def try_json(text: str) -> bool:
    text = text.strip()
    for candidate in (text, text[text.find("{"):] if "{" in text else ""):
        if not candidate:
            continue
        try:
            json.loads(candidate)
            return True
        except (json.JSONDecodeError, ValueError):
            continue
    return False


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
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[100, 200, 300, 400])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=CHECKPOINT_ROOT,
        help="e.g. outputs/dpo-v2-full for the v2 sweep (ADR-007 Stage 4)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
        help="e.g. eval/out/dpo_v2_sweep for the v2 sweep",
    )
    args = parser.parse_args()

    load_dotenv()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = load_eval_prompts(tokenizer, args.num_prompts)
    print(f"Sweeping on {len(prompts)} held-out eval prompts")

    print(f"Loading base model: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto"
    )

    print(f"Attaching SFT adapter: {SFT_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base, str(SFT_ADAPTER_DIR), adapter_name="sft")

    candidates = ["sft"]
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
        for i, ex in enumerate(prompts):
            gen = generate(model, tokenizer, ex["prompt"], args.max_new_tokens)
            rows.append(
                {
                    "prompt_idx": i,
                    "candidate": cand,
                    "generation": gen,
                    "chosen": ex["chosen"],
                    "exact_match": gen.strip() == ex["chosen"].strip(),
                    "json_valid": try_json(gen),
                }
            )
            print(f"  [{i + 1}/{len(prompts)}] match={rows[-1]['exact_match']} json={rows[-1]['json_valid']}")

    gen_path = args.out_dir / "generations.jsonl"
    with gen_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {gen_path}")

    # Summary + side-by-side report
    lines = ["# DPO checkpoint sweep report", ""]
    lines += ["| candidate | exact match | json valid |", "|---|---|---|"]
    for cand in candidates:
        sub = [r for r in rows if r["candidate"] == cand]
        em = sum(r["exact_match"] for r in sub)
        jv = sum(r["json_valid"] for r in sub)
        lines.append(f"| {cand} | {em}/{len(sub)} | {jv}/{len(sub)} |")
    lines.append("")

    for i in range(len(prompts)):
        lines.append(f"## Prompt {i}")
        lines.append("")
        lines.append("**Chosen (reference):**")
        lines.append("```\n" + prompts[i]["chosen"] + "\n```")
        for cand in candidates:
            row = next(r for r in rows if r["prompt_idx"] == i and r["candidate"] == cand)
            flag = "MATCH" if row["exact_match"] else "diff"
            lines.append(f"**{cand}** ({flag}):")
            lines.append("```\n" + row["generation"] + "\n```")
        lines.append("")

    report_path = args.out_dir / "sweep_report.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path}")

    print("\n=== Sweep summary ===")
    for line in lines[2 : 4 + len(candidates)]:
        print(line)


if __name__ == "__main__":
    main()
