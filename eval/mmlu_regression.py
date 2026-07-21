"""MMLU 5-shot regression check: base vs SFT vs DPO checkpoint sweep.

Wrapper around EleutherAI's lm-evaluation-harness. Runs the standard 5-shot
`mmlu` task against the base Llama-3.1-8B-Instruct plus our SFT adapter and
DPO v2 checkpoints, then aggregates per-candidate MMLU accuracy into a single
report. The interview-defensible claim ("DPO did not collapse general
capability") requires numbers directly comparable to published Llama-3.1
baselines, which is why we shell out to the standard harness rather than
rolling our own MMLU runner.

Each candidate is evaluated sequentially — the harness is GPU-bound and this
must run in its own tmux session after any concurrent GPU job completes.

Writes:

- `eval/out/mmlu_regression/<candidate>/results*.json` — raw lm_eval output
- `eval/out/mmlu_regression/report.md` — summary table

Usage:
    python eval/mmlu_regression.py
    python eval/mmlu_regression.py --checkpoints 50 150
    python eval/mmlu_regression.py --base-only
    python eval/mmlu_regression.py --skip-base
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SFT_ADAPTER_DIR = Path("./outputs/sft-full")
DEFAULT_CHECKPOINT_ROOT = Path("./outputs/dpo-v2-full")
OUT_DIR = Path("./eval/out/mmlu_regression")
MMLU_TASK = "mmlu"
NUM_FEWSHOT = 5

LM_EVAL_VERSION = "0.4.3"


def check_or_install_lm_eval() -> None:
    """Ensure lm_eval is importable; pip-install pinned version if missing."""
    try:
        import lm_eval  # noqa: F401
        return
    except ImportError:
        pass

    print(f"lm_eval not found; installing lm-eval[hf]=={LM_EVAL_VERSION} ...")
    try:
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                f"lm-eval[hf]=={LM_EVAL_VERSION}",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(
            "\nFAILED to install lm-evaluation-harness. "
            "Install manually and rerun:\n"
            "  https://github.com/EleutherAI/lm-evaluation-harness\n"
            f"  pip install 'lm-eval[hf]=={LM_EVAL_VERSION}'\n"
            f"Underlying error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import lm_eval  # noqa: F401
    except ImportError as e:
        print(f"lm_eval still not importable after install: {e}", file=sys.stderr)
        sys.exit(1)


def run_lm_eval(candidate: str, out_subdir: Path, peft_dir: Path | None,
                batch_size: str) -> None:
    """Invoke `lm_eval` CLI for a single candidate."""
    out_subdir.mkdir(parents=True, exist_ok=True)

    model_args = f"pretrained={BASE_MODEL},dtype=bfloat16"
    if peft_dir is not None:
        # lm-eval's HF wrapper accepts a `peft=` kwarg and loads the adapter
        # via PeftModel.from_pretrained on top of the base `pretrained` model.
        # No special flag needed for Llama-3.1 — standard PEFT LoRA loading.
        model_args += f",peft={peft_dir}"

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", MMLU_TASK,
        "--num_fewshot", str(NUM_FEWSHOT),
        "--batch_size", batch_size,
        "--output_path", str(out_subdir),
    ]
    print(f"\n=== Running lm_eval for {candidate} ===")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_results_json(out_subdir: Path) -> Path:
    """lm-eval writes to either results.json or a timestamped subdirectory
    (e.g. <out>/<model_sanitized>/results_<timestamp>.json). Search recursively
    and pick the most recently modified match."""
    matches = sorted(
        out_subdir.rglob("results*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No results*.json found under {out_subdir} — did lm_eval fail?"
        )
    return matches[0]


def extract_mmlu_score(results_json: Path) -> float:
    """Pull the aggregate MMLU accuracy from an lm-eval results.json.

    lm-eval reports the aggregate under results["mmlu"]["acc,none"] (with
    stderr under "acc_stderr,none"). Fall back to any key starting with "acc"
    for older/newer harness versions.
    """
    with results_json.open() as f:
        data = json.load(f)
    results = data.get("results", {})
    if MMLU_TASK not in results:
        raise KeyError(
            f"'mmlu' aggregate missing from {results_json}; "
            f"top-level keys: {list(results.keys())[:5]}"
        )
    mmlu = results[MMLU_TASK]
    for key in ("acc,none", "acc"):
        if key in mmlu:
            return float(mmlu[key])
    for k, v in mmlu.items():
        if k.startswith("acc") and not k.startswith("acc_stderr") \
                and isinstance(v, (int, float)):
            return float(v)
    raise KeyError(f"No acc field in mmlu results: {list(mmlu.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=int, nargs="+",
                        default=[50, 100, 150])
    parser.add_argument("--checkpoint-root", type=Path,
                        default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--batch-size", type=str, default="auto",
                        help="Passed through to lm_eval --batch_size")
    parser.add_argument("--base-only", action="store_true",
                        help="Run only the base Llama-3.1-8B-Instruct baseline")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip base (e.g. already run in a previous invocation)")
    args = parser.parse_args()

    load_dotenv()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    check_or_install_lm_eval()

    # Build the candidate list. Order matters: base first so the delta column
    # in the report has a reference.
    candidates: list[tuple[str, Path | None]] = []
    if not args.skip_base:
        candidates.append(("base", None))
    if not args.base_only:
        if not SFT_ADAPTER_DIR.exists():
            raise FileNotFoundError(f"Missing SFT adapter dir: {SFT_ADAPTER_DIR}")
        candidates.append(("sft", SFT_ADAPTER_DIR))
        for step in args.checkpoints:
            ckpt_dir = args.checkpoint_root / f"checkpoint-{step}"
            if not ckpt_dir.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_dir}")
            candidates.append((f"dpo-{step}", ckpt_dir))

    if not candidates:
        print("Nothing to run (base skipped and --base-only not set).")
        sys.exit(0)

    # Run each candidate sequentially (GPU-bound).
    for tag, peft_dir in candidates:
        run_lm_eval(tag, args.out_dir / tag, peft_dir, args.batch_size)

    # Aggregate.
    scores: dict[str, float] = {}
    for tag, _ in candidates:
        results_json = find_results_json(args.out_dir / tag)
        scores[tag] = extract_mmlu_score(results_json)
        print(f"  {tag}: mmlu={scores[tag]:.4f}  ({results_json})")

    base_score = scores.get("base")
    lines = ["# MMLU 5-shot regression — base vs SFT vs DPO sweep", ""]
    lines += [
        "| candidate | mmlu (5-shot) | delta vs base |",
        "|---|---|---|",
    ]
    for tag, _ in candidates:
        score = scores[tag]
        if tag == "base" or base_score is None:
            delta = "—"
        else:
            d = score - base_score
            delta = f"{d:+.4f}"
        lines.append(f"| {tag} | {score:.4f} | {delta} |")
    lines.append("")

    report_path = args.out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nWrote {report_path}")

    print("\n=== MMLU regression summary ===")
    for line in lines[2 : 4 + len(candidates)]:
        print(line)


if __name__ == "__main__":
    main()
