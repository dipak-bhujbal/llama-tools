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
- `eval/out/bfcl_simple/run_manifest.json` — provenance: code/inputs/environment,
  plus a "running" -> "complete"/"incomplete" lifecycle so a run that dies
  mid-generation still leaves a record of what it was and why it stopped

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
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from bfcl_category_config import SUPPORTED_CATEGORIES, resolve_category_paths
from bfcl_scoring import preflight_key_names, score
from dev_subset_gate import load_dev_subset_ids, restrict_to_dev_subset
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
SFT_ADAPTER_DIR = Path("./outputs/sft-full")
CHECKPOINT_ROOT = Path("./outputs/dpo-v2-full")
REPO_ROOT = Path(__file__).resolve().parent.parent

# The three files that together describe one run's evidence. All three are
# refused-if-nonempty and all three are removed by --overwrite: a directory
# holding a mix of old and new siblings can describe a run that never
# actually happened.
SIBLING_FILENAMES = ("generations.jsonl", "report.md", "run_manifest.json")


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


def _package_versions() -> dict:
    """Best-effort installed-version snapshot for the libraries that shape
    generation behaviour. A silent dependency upgrade (e.g. transformers
    changing generate() defaults) is exactly the kind of confound a
    provenance record exists to catch; tolerate a missing package rather than
    crash the run over an optional-dependency gap.
    """
    versions = {}
    for pkg in ("transformers", "peft", "accelerate", "torch", "huggingface_hub"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def build_environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "package_versions": _package_versions(),
    }


def build_initial_manifest(args, candidates: list, category_paths, n_prompts: int) -> dict:
    """Everything knowable before the first token is generated.

    Written to disk immediately (status "running"), before model loading or
    generation starts, so a run killed by a wall-clock timeout still leaves a
    record of what it *was going to do* — code, weights, inputs, environment —
    even though it never gets to record what happened. See _finalize_manifest
    for the completion/failure side of this lifecycle.
    """
    return {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
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
        "checkpoint_root": str(args.checkpoint_root),
        "checkpoints": args.checkpoints,
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
        "environment": build_environment(),
    }


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write via a temp file in the same directory + os.replace.

    os.replace is atomic on POSIX and on Windows (Python emulates POSIX
    rename semantics there), so any concurrent reader — or a process that
    crashes mid-write — always sees either the previous complete manifest or
    the new complete one, never a half-written, unparseable one.
    """
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, path)


def write_run_manifest(out_dir: Path, manifest: dict) -> Path:
    """Persist the manifest at its current lifecycle stage.

    Called once before the first generation (status "running") and once more
    at the very end (status "complete" or "incomplete") — the atomic write
    means neither call can ever leave a corrupt file on disk.
    """
    path = out_dir / "run_manifest.json"
    _atomic_write_json(path, manifest)
    return path


def _finalize_manifest(
    manifest: dict,
    *,
    status: str,
    started_monotonic: float,
    rows_written: int,
    gen_path: Path,
    report_path: Path,
    failure_reason: str | None,
    validation: dict | None,
) -> None:
    """Fill in the fields only knowable once the run has ended (or died).

    Shared by the success and failure paths so neither one can forget a field
    the other sets. Mutates `manifest` in place; the caller is responsible for
    writing it to disk (via write_run_manifest) afterward.
    """
    manifest["status"] = status
    manifest["ended_utc"] = datetime.now(UTC).isoformat()
    manifest["elapsed_seconds"] = time.monotonic() - started_monotonic
    manifest["rows_written"] = rows_written
    outputs = {}
    if gen_path.exists():
        outputs["generations"] = {"path": str(gen_path), "sha256": _sha256_file(gen_path)}
    if report_path.exists():
        outputs["report"] = {"path": str(report_path), "sha256": _sha256_file(report_path)}
    manifest["outputs"] = outputs
    if failure_reason is not None:
        manifest["failure_reason"] = failure_reason
    if validation is not None:
        manifest["validation"] = validation


class _Terminated(SystemExit):
    """Raised from the SIGTERM handler installed in main().

    A bare SIGTERM kills the process immediately with no chance to run
    cleanup code; routing it through this exception instead lets it flow
    through the same try/except/finally as any other failure, so the
    manifest still gets finalized to "incomplete". This is precisely the
    wall-clock-timeout scenario the whole manifest lifecycle exists to cover
    — a scheduler (Slurm, k8s preemption, a plain `timeout` wrapper) sends
    SIGTERM, not a Python exception, when a job's time budget runs out.
    """


def _raise_on_sigterm(signum, frame) -> None:
    raise _Terminated(f"received signal {signum}")


def _install_sigterm_handler() -> None:
    signal.signal(signal.SIGTERM, _raise_on_sigterm)


def validate_run_outputs(out_dir: Path, expected_ids, candidates: list) -> dict:
    """Re-read generations.jsonl FROM DISK (not the in-memory `rows`) and
    check it actually contains what a complete run promises.

    This runs after the in-memory completeness checks so that a bug in the
    in-memory bookkeeping (e.g. a row appended to `rows` but the write to
    disk silently failing) cannot masquerade as a complete, correct run —
    the manifest's "complete" status must be backed by what is actually on
    disk, since that is the only artifact a downstream analysis ever reads.

    Checks, independently:
    - total row count == len(expected_ids) * len(candidates)
    - the set of (id, model_name) pairs exactly equals the cartesian product
    - no (id, model_name) pair appears more than once
    - every line parses as JSON

    A duplicate pair can mask a missing pair while keeping the total row
    count correct, which is why these are checked separately rather than
    collapsed into a single count comparison.
    """
    gen_path = out_dir / "generations.jsonl"
    expected_pairs = {(pid, cand) for pid in expected_ids for cand in candidates}

    result = {
        "ok": False,
        "rows_on_disk": 0,
        "expected_rows": len(expected_pairs),
        "unparseable_lines": 0,
        "duplicate_pairs": [],
        "missing_pairs": [],
        "extra_pairs": [],
    }
    if not gen_path.exists():
        result["unparseable_lines"] = None
        return result

    total_lines = 0
    unparseable = 0
    seen_set: set = set()
    duplicates: set = set()
    with gen_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                unparseable += 1
                continue
            pair = (row.get("id"), row.get("model_name"))
            if pair in seen_set:
                duplicates.add(pair)
            seen_set.add(pair)

    missing = expected_pairs - seen_set
    extra = seen_set - expected_pairs

    result["rows_on_disk"] = total_lines
    result["unparseable_lines"] = unparseable
    result["duplicate_pairs"] = sorted(duplicates)
    result["missing_pairs"] = sorted(missing)
    result["extra_pairs"] = sorted(extra)
    result["ok"] = (
        unparseable == 0
        and total_lines == len(expected_pairs)
        and not duplicates
        and not missing
        and not extra
    )
    return result


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
    # Installed first: a wall-clock timeout can arrive at any point below,
    # and every point below must be covered by the try/except/finally that
    # finalizes the manifest.
    _install_sigterm_handler()

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

    gen_path = args.out_dir / "generations.jsonl"
    report_path = args.out_dir / "report.md"
    sibling_paths = [args.out_dir / name for name in SIBLING_FILENAMES]

    # Refuse to write into a directory that already holds ANY evidence file,
    # not just generations.jsonl. Truncating only generations.jsonl under
    # --overwrite used to leave a stale report.md / run_manifest.json behind,
    # so the directory could describe a run that no longer exists — a paid
    # run's provenance must never be a mix of two different runs.
    stale = [p for p in sibling_paths if p.exists() and p.stat().st_size > 0]
    if stale and not args.overwrite:
        names = ", ".join(p.name for p in stale)
        raise SystemExit(
            f"refusing to write into {args.out_dir}: non-empty {names} present\n"
            f"use a fresh --out-dir, or pass --overwrite if you truly mean to discard it"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for p in sibling_paths:
            if p.exists():
                p.unlink()

    print(f"Loading BFCL prompts: {category_paths.questions}")
    prompts_raw = load_jsonl(category_paths.questions)
    gt_raw = load_jsonl(category_paths.answer_key)

    if category_paths.is_development_subset:
        # Only the pinned 258 are ever scored. Every kill line and eligibility
        # test is denominated in items against exactly this set, so scoring the
        # full 1,053-row parent — or any other subset — would silently move every
        # threshold that referenced the frozen baseline (prereg §3.2, §3.3).
        #
        # Both sides are restricted, not just the questions: §3.2 requires the
        # runner to match "the same 258 answer rows", and the id reconciliation
        # below compares the two sets for equality.
        if args.num_prompts is not None:
            raise SystemExit(
                "--num-prompts cannot be combined with the development set: the "
                "subset is pinned at 258 items and a partial run is not comparable "
                "to the frozen baseline"
            )
        prompts_raw, gt_raw = restrict_to_dev_subset(
            prompts_raw, gt_raw, load_dev_subset_ids(REPO_ROOT)
        )
        print(f"Development subset gate: {len(prompts_raw)} pinned items")

    gt_by_id = {r["id"]: r["ground_truth"][0] for r in gt_raw}
    prompt_ids = {row["id"] for row in prompts_raw}
    if prompt_ids != set(gt_by_id):
        missing_keys = sorted(prompt_ids - set(gt_by_id))
        extra_keys = sorted(set(gt_by_id) - prompt_ids)
        raise ValueError(
            f"question/answer id mismatch: missing={missing_keys[:5]} "
            f"extra={extra_keys[:5]}"
        )

    # Exact name matching is only fair while the key's name is among the tools
    # the item presented; where it is not, the item is unpassable and the model
    # is blamed for a benchmark defect. Checked here, before any model loads,
    # so a defective key costs no GPU time.
    checked = preflight_key_names(
        {row["id"]: row for row in prompts_raw}, {r["id"]: r for r in gt_raw}
    )
    print(f"Answer-key preflight passed: {checked} items expect only presented tool names")

    if args.num_prompts is not None:
        prompts_raw = prompts_raw[: args.num_prompts]
    print(f"Evaluating on {len(prompts_raw)} prompts")

    # "base" is the unmodified base model: same weights, adapters disabled at
    # generation time. It is not an adapter, so it is never passed to
    # set_adapter(). Computed here, before any model is loaded, because the
    # candidate list is pure args-derived and the initial manifest needs it.
    candidates = build_candidate_names(
        include_base=args.include_base,
        sft_only=args.sft_only,
        checkpoints=args.checkpoints,
    )

    # Written BEFORE the first generation: if a wall-clock timeout kills this
    # run at any point after this, the rows written so far otherwise have no
    # provenance at all. See _finalize_manifest for the other end of this
    # lifecycle.
    manifest = build_initial_manifest(args, candidates, category_paths, len(prompts_raw))
    write_run_manifest(args.out_dir, manifest)

    run_status = "incomplete"  # pessimistic default; only a clean finish flips this
    failure_reason: str | None = None
    validation: dict | None = None
    started_monotonic = time.monotonic()
    rows: list[dict] = []
    gen_file = None

    try:
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

        report_path.write_text("\n".join(lines))
        print(f"Wrote {report_path}")

        # On-disk validation: re-read generations.jsonl (not the in-memory
        # `rows`) so a bug in the in-memory bookkeeping cannot masquerade as a
        # complete, correct run. Recorded in the manifest either way — see
        # _finalize_manifest below.
        expected_ids = {ex["id"] for ex in built_prompts}
        validation = validate_run_outputs(args.out_dir, expected_ids, candidates)
        if not validation["ok"]:
            raise SystemExit(f"validate_run_outputs failed: {validation}")

        run_status = "complete"

        print(f"\n=== BFCL {args.category} summary ===")
        for line in lines[2 : 4 + len(candidates)]:
            print(line)

    except BaseException as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if gen_file is not None and not gen_file.closed:
            gen_file.close()
        # An exception raised in a finally block REPLACES the exception that is
        # already propagating, so a failure to write the manifest would hide
        # the real cause of the run's death. Report the secondary failure
        # loudly and let the original propagate — losing the manifest is bad,
        # losing the reason the run died is worse.
        try:
            _finalize_manifest(
                manifest,
                status=run_status,
                started_monotonic=started_monotonic,
                rows_written=len(rows),
                gen_path=gen_path,
                report_path=report_path,
                failure_reason=failure_reason,
                validation=validation,
            )
            manifest_path = write_run_manifest(args.out_dir, manifest)
            print(f"Wrote {manifest_path} (status={manifest['status']})")
        except Exception as manifest_exc:
            print(
                f"WARNING: failed to write run manifest: "
                f"{type(manifest_exc).__name__}: {manifest_exc}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
