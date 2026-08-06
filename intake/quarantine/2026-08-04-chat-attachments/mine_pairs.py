#!/usr/bin/env python3
"""On-policy DPO pair mining pilot (llama-tools rescue study, step 2).

Pipeline per the study playbook:
  1. --self-test : unit-test the verifier against the synthetic fixture set
                   (fixture_pairs_*.jsonl). Run this FIRST. Mining refuses to
                   run unless the self-test passes in the same invocation or
                   --skip-self-test is given explicitly.
  2. mine        : sample the shipped SFT checkpoint 8x per prompt at temp 0.8
                   over prompts drawn from the training pool, verify each
                   generation programmatically, and bucket prompts:
                     8/8 correct -> discarded (model has mastered it)
                     0/8 correct -> sft_bucket.jsonl (rejection-sampling SFT)
                     1-7 of 8    -> mined_pairs.jsonl (on-policy DPO pairs)
  3. decide      : print the pass-rate histogram and apply the pre-registered
                   thresholds (>=1000 projected pairs: proceed; 300-1000:
                   proceed cautiously; <300: do not run DPO).

Resume & rollback: every processed prompt is recorded in <out-dir>/ledger.jsonl
and all outputs are written incrementally, so an interrupted run resumes where
it left off (just rerun the same command). Use --redo-last N to roll back the
most recent N prompts (e.g. after an interruption mid-batch or a suspicious
result) and --fresh to start the out-dir over.

All provenance in the output meta is MEASURED (real pass rates, real verifier,
real sampling params), unlike the synthetic fixture set.

Usage:
  python mine_pairs.py --self-test --fixtures ./fixtures
  python mine_pairs.py --adapter /path/or/hf-id/of/sft-adapter \
      --fixtures ./fixtures --n-prompts 1000 --out-dir mining_out
  # quick read first:
  python mine_pairs.py --adapter ... --fixtures ./fixtures --n-prompts 100

Requires (mining only): torch, transformers, peft, datasets. The self-test is
pure stdlib.
"""
import argparse
import collections
import datetime
import glob
import hashlib
import json
import os
import random
import re
import sys

VERIFIER_VERSION = "onpolicy_verifier_v1"

# ----------------------------------------------------------------- verifier -

TOOLCALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _first_balanced_json(text):
    """Return the first balanced {...} substring, or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def parse_tool_call(text):
    """Parse a generation into a tool call dict {'name','arguments'} or None.

    Accepts either <tool_call>{...}</tool_call> (Hermes style) or a bare JSON
    object. Returns (call, reason) where call is None on failure.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "empty"
    m = TOOLCALL_TAG_RE.search(text)
    candidate = m.group(1) if m else _first_balanced_json(text)
    if candidate is None:
        return None, "no_json_object"
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None, "malformed_json"
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None, "not_a_tool_call"
    args = obj.get("arguments", {})
    if isinstance(args, str):
        # some models emit arguments as a JSON string
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None, "malformed_arguments"
    if not isinstance(args, dict):
        return None, "arguments_not_object"
    return {"name": obj["name"], "arguments": args}, None


def extract_tools_from_system(system_text):
    """Pull the tool schema list out of a system prompt.

    Handles both '<tools>[...]</tools>' (Hermes) and 'tools: [...]' inline
    JSON (fixture style). Returns {tool_name: params_dict} or None.
    """
    m = re.search(r"<tools>\s*(\[.*?\])\s*</tools>", system_text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        start = system_text.find("[")
        if start != -1:
            depth = 0
            for i in range(start, len(system_text)):
                if system_text[i] == "[":
                    depth += 1
                elif system_text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        blob = system_text[start:i + 1]
                        break
    if blob is None:
        return None
    try:
        tools = json.loads(blob)
    except json.JSONDecodeError:
        return None
    out = {}
    for t in tools if isinstance(tools, list) else []:
        if not isinstance(t, dict):
            continue
        name = t.get("name") or (t.get("function") or {}).get("name")
        params = t.get("parameters") or (t.get("function") or {}).get("parameters")
        if isinstance(params, dict) and "properties" in params:
            props = params.get("properties", {})
            required = set(params.get("required", []))
            params = {k: dict(v, required=(k in required))
                      for k, v in props.items() if isinstance(v, dict)}
        if isinstance(name, str) and isinstance(params, dict):
            out[name] = params
    return out or None


def _norm(v):
    if isinstance(v, str):
        s = v.strip()
        try:
            return float(s)
        except ValueError:
            return s
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def verify(generation_text, gt):
    """Verify one generation against ground truth.

    gt: {"type": "no_call"} or
        {"type": "call", "name": str, "arguments": dict, "schema": dict|None}

    Returns (verdict, reason) with verdict in {"pass", "fail", "ambiguous"}.
    Ambiguous = a declared optional parameter present in the generation but
    absent from ground truth: we cannot verify its value, so the generation is
    excluded from BOTH chosen and rejected candidates (audit-guardrail rule
    against punishing acceptable-but-different outputs).
    """
    call, why = parse_tool_call(generation_text)
    if gt["type"] == "no_call":
        if call is not None:
            return "fail", "spurious_tool_call"
        if why == "empty":
            return "fail", "empty"
        return "pass", None
    # ground truth is a call
    if call is None:
        if why in ("malformed_json", "malformed_arguments"):
            return "fail", "malformed_syntax"
        return "fail", "missed_tool_call"
    if call["name"] != gt["name"]:
        return "fail", "wrong_function_selection"
    schema = gt.get("schema")
    decl = schema.get(gt["name"]) if schema else None
    gen_args, gt_args = call["arguments"], gt["arguments"]
    if decl is not None:
        for k in gen_args:
            if k not in decl:
                return "fail", "hallucinated_parameter"
    for k in gt_args:
        if k not in gen_args:
            return "fail", "missing_required_parameter"
    for k in gt_args:
        if _norm(gen_args[k]) != _norm(gt_args[k]):
            return "fail", "wrong_param_value"
    extras = [k for k in gen_args if k not in gt_args]
    if extras:
        if decl is None:
            return "ambiguous", "extra_param_unknown_schema"
        return "ambiguous", "extra_optional_param"
    return "pass", None


# ---------------------------------------------------------------- self-test -

def self_test(fixtures_dir):
    """Run the verifier against the synthetic fixture set. Returns True/False.

    For every fixture pair: chosen must verify as pass, rejected must verify
    as fail (not ambiguous), with the fail reason matching the labeled
    error_type where the mapping is 1:1.
    """
    files = sorted(glob.glob(os.path.join(fixtures_dir, "fixture_pairs_*.jsonl")))
    if not files:
        print(f"[self-test] no fixture_pairs_*.jsonl under {fixtures_dir}", file=sys.stderr)
        return False
    total = chosen_bad = rejected_bad = reason_mismatch = 0
    per_type = collections.Counter()
    caught = collections.Counter()
    for path in files:
        for line in open(path):
            row = json.loads(line)
            total += 1
            et = row["meta"]["error_type"]
            per_type[et] += 1
            sys_text = row["prompt"][0]["content"]
            chosen = row["chosen"][0]["content"]
            rejected = row["rejected"][0]["content"]
            gt_call, _ = parse_tool_call(chosen)
            if gt_call is None:
                gt = {"type": "no_call"}
            else:
                gt = {"type": "call", "name": gt_call["name"],
                      "arguments": gt_call["arguments"],
                      "schema": extract_tools_from_system(sys_text)}
            v_c, _ = verify(chosen, gt)
            v_r, r_r = verify(rejected, gt)
            if v_c != "pass":
                chosen_bad += 1
            if v_r != "fail":
                rejected_bad += 1
            else:
                caught[et] += 1
                if et != r_r and et != "malformed_syntax":
                    reason_mismatch += 1
                elif et == "malformed_syntax" and r_r not in (
                        "malformed_syntax", "missed_tool_call"):
                    reason_mismatch += 1
    print(f"[self-test] {total} fixture pairs from {len(files)} file(s)")
    for et in sorted(per_type):
        print(f"  {et:28s} caught {caught[et]}/{per_type[et]}")
    print(f"  chosen misjudged: {chosen_bad} | rejected missed: {rejected_bad}"
          f" | reason mismatches: {reason_mismatch}")
    ok = chosen_bad == 0 and rejected_bad == 0
    print(f"[self-test] {'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------ hermes loader -

def load_prompt_pool(dataset_name, dataset_config, n_prompts, seed):
    """Load single-turn prompts + ground truth from a Hermes-style dataset.

    Yields dicts: {"messages": [system,user], "gt": gt, "source": str}.
    Skips rows it cannot confidently parse; dedupes on user text.
    """
    from datasets import load_dataset  # heavy import kept local
    try:
        ds = load_dataset(dataset_name, dataset_config, split="train")
    except Exception:
        ds = load_dataset(dataset_name, split="train")
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    seen_users = set()
    out = []
    for idx in order:
        if len(out) >= n_prompts:
            break
        row = ds[idx]
        convs = row.get("conversations") or row.get("messages")
        if not convs:
            continue
        def get(field, item):
            return item.get(field) or item.get({"from": "role",
                                                "value": "content"}[field], "")
        sys_t = next((c.get("value") or c.get("content", "") for c in convs
                      if (c.get("from") or c.get("role")) == "system"), None)
        hum = next((c.get("value") or c.get("content", "") for c in convs
                    if (c.get("from") or c.get("role")) in ("human", "user")), None)
        gpt = next((c.get("value") or c.get("content", "") for c in convs
                    if (c.get("from") or c.get("role")) in ("gpt", "assistant")), None)
        if not sys_t or not hum or not gpt:
            continue
        if hum in seen_users:
            continue
        gt_call, _ = parse_tool_call(gpt)
        if gt_call is None:
            gt = {"type": "no_call"}
        else:
            gt = {"type": "call", "name": gt_call["name"],
                  "arguments": gt_call["arguments"],
                  "schema": extract_tools_from_system(sys_t)}
        seen_users.add(hum)
        out.append({"messages": [{"role": "system", "content": sys_t},
                                 {"role": "user", "content": hum}],
                    "gt": gt,
                    "source": f"{dataset_name}#{idx}"})
    return out


# ---------------------------------------------------------- decontamination -

def build_bfcl_ngrams(bfcl_paths, n=13):
    grams = set()
    funcsigs = set()
    for path in bfcl_paths:
        for line in open(path):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = row.get("question")
            if isinstance(q, list):
                q = " ".join(str(x) for x in q)
            if isinstance(q, str):
                toks = q.lower().split()
                for i in range(len(toks) - n + 1):
                    grams.add(tuple(toks[i:i + n]))
            for f in row.get("function", []) if isinstance(row.get("function"), list) else []:
                name = f.get("name")
                params = f.get("parameters", {})
                props = sorted((params.get("properties") or {}).keys())
                if name:
                    funcsigs.add((name, tuple(props)))
    return grams, funcsigs


def is_contaminated(prompt_item, grams, funcsigs, n=13):
    user = prompt_item["messages"][1]["content"].lower().split()
    for i in range(len(user) - n + 1):
        if tuple(user[i:i + n]) in grams:
            return True
    gt = prompt_item["gt"]
    if gt["type"] == "call" and gt.get("schema"):
        for name, params in gt["schema"].items():
            if (name, tuple(sorted(params.keys()))) in funcsigs:
                return True
    return False


# ------------------------------------------------------------------ resume -

def _append_jsonl(path, row):
    """Append one row and force it to disk so an interruption loses at most
    the row being written."""
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_ledger(out_dir):
    path = os.path.join(out_dir, "ledger.jsonl")
    rows = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # torn tail line from an interrupted write; re-mined
    return rows


def _rewrite_without_sources(path, sources, source_in_meta):
    if not os.path.exists(path):
        return
    kept = []
    for line in open(path):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        src = (row.get("meta", {}) if source_in_meta else row).get("source")
        if src not in sources:
            kept.append(row)
    with open(path, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")


def redo_last(out_dir, n):
    """Roll back the last n processed prompts (ledger + all output rows) so
    the next mining run re-mines them."""
    ledger = load_ledger(out_dir)
    if not ledger:
        print("[redo] ledger is empty; nothing to roll back")
        return
    n = min(n, len(ledger))
    rolled = ledger[-n:]
    targets = {r["source"] for r in rolled}
    with open(os.path.join(out_dir, "ledger.jsonl"), "w") as f:
        for r in ledger[:-n]:
            f.write(json.dumps(r) + "\n")
    _rewrite_without_sources(os.path.join(out_dir, "mined_pairs.jsonl"),
                             targets, source_in_meta=True)
    _rewrite_without_sources(os.path.join(out_dir, "sft_bucket.jsonl"),
                             targets, source_in_meta=False)
    _rewrite_without_sources(os.path.join(out_dir, "ambiguous_review.jsonl"),
                             targets, source_in_meta=False)
    print(f"[redo] rolled back {n} prompt(s); they will be re-mined next run:")
    for r in rolled:
        print(f"  - {r['source']} (was: {r['outcome']}, "
              f"{r['n_pass']}/{r.get('samples', '?')} pass)")


# ------------------------------------------------------------------- mining -

def run_mining(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch.manual_seed(args.seed)
    print(f"[mine] loading base {args.base} + adapter {args.adapter}")
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    print(f"[mine] loading prompt pool from {args.dataset} ({args.dataset_config})")
    pool = load_prompt_pool(args.dataset, args.dataset_config,
                            args.n_prompts, args.seed)
    print(f"[mine] pool size after parse+dedup: {len(pool)}")

    if args.bfcl_path:
        grams, funcsigs = build_bfcl_ngrams(args.bfcl_path)
        before = len(pool)
        pool = [p for p in pool if not is_contaminated(p, grams, funcsigs)]
        print(f"[mine] decontamination: dropped {before - len(pool)} prompt(s)")
    else:
        print("[mine] WARNING: no --bfcl-path given; n-gram decontamination "
              "skipped. Function-schema overlap with BFCL not checked.")

    os.makedirs(args.out_dir, exist_ok=True)
    ledger_path = os.path.join(args.out_dir, "ledger.jsonl")
    pairs_path = os.path.join(args.out_dir, "mined_pairs.jsonl")
    sft_path = os.path.join(args.out_dir, "sft_bucket.jsonl")
    ambig_path = os.path.join(args.out_dir, "ambiguous_review.jsonl")

    ledger = load_ledger(args.out_dir)
    done = {r["source"] for r in ledger}
    if done:
        pool_before = len(pool)
        pool = [p for p in pool if p["source"] not in done]
        print(f"[mine] RESUME: {len(done)} prompt(s) already in the ledger; "
              f"{len(pool)} of {pool_before} remaining")
    prior = collections.Counter(r["outcome"] for r in ledger)
    n_pairs_total = prior.get("pair", 0)
    n_sft_total = prior.get("sft_bucket", 0)

    for start in range(0, len(pool), args.batch_size):
        batch = pool[start:start + args.batch_size]
        prompts = [tok.apply_chat_template(p["messages"], tokenize=False,
                                           add_generation_prompt=True)
                   for p in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, do_sample=True, temperature=args.temperature,
                top_p=args.top_p, num_return_sequences=args.samples,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)

        for bi, item in enumerate(batch):
            outs = texts[bi * args.samples:(bi + 1) * args.samples]
            verdicts = [verify(t, item["gt"]) for t in outs]
            n_pass = sum(v == "pass" for v, _ in verdicts)
            n_ambig = sum(v == "ambiguous" for v, _ in verdicts)
            for t, (v, r) in zip(outs, verdicts):
                if v == "ambiguous":
                    _append_jsonl(ambig_path, {
                        "prompt": item["messages"], "generation": t,
                        "reason": r, "source": item["source"]})
            outcome, error_type = "discarded_mastered", None
            if n_pass == 0:
                _append_jsonl(sft_path, {
                    "messages": item["messages"], "gt": item["gt"],
                    "source": item["source"],
                    "sample_verdicts": [r for _, r in verdicts]})
                outcome = "sft_bucket"
                n_sft_total += 1
            elif n_pass < args.samples:
                rng = random.Random(hash(item["source"]) & 0xFFFFFFFF)
                chosen_pool = [t for t, (v, _) in zip(outs, verdicts)
                               if v == "pass"]
                fail_pool = [(t, r) for t, (v, r) in zip(outs, verdicts)
                             if v == "fail"]
                if not fail_pool:
                    outcome = "discarded_no_verified_fail"
                else:
                    non_mal = [(t, r) for t, r in fail_pool
                               if r != "malformed_syntax"]
                    rej_text, rej_reason = rng.choice(non_mal or fail_pool)
                    cho_text = rng.choice(chosen_pool)
                    gap = abs(len(cho_text) - len(rej_text)) / \
                        max(len(cho_text), len(rej_text), 1)
                    if gap > 0.40 and rej_reason not in ("missed_tool_call",
                                                         "spurious_tool_call"):
                        outcome = "discarded_similarity_floor"
                    else:
                        _append_jsonl(pairs_path, {
                            "prompt": item["messages"],
                            "chosen": [{"role": "assistant",
                                        "content": cho_text}],
                            "rejected": [{"role": "assistant",
                                          "content": rej_text}],
                            "meta": {
                                "pair_id": hashlib.sha1(
                                    item["source"].encode()).hexdigest()[:12],
                                "error_type": rej_reason,
                                "pass_rate": f"{n_pass}/{args.samples}",
                                "ambiguous_count": n_ambig,
                                "on_policy": True,
                                "gen": {"base": args.base,
                                        "adapter": args.adapter,
                                        "temperature": args.temperature,
                                        "top_p": args.top_p,
                                        "seed": args.seed,
                                        "max_new_tokens": args.max_new_tokens},
                                "verified_by": VERIFIER_VERSION,
                                "self_tested_against_fixtures": True,
                                "source": item["source"],
                                "mined_at": datetime.date.today().isoformat()}})
                        outcome, error_type = "pair", rej_reason
                        n_pairs_total += 1
            _append_jsonl(ledger_path, {
                "source": item["source"], "n_pass": n_pass,
                "n_ambiguous": n_ambig, "samples": args.samples,
                "outcome": outcome, "error_type": error_type})
        done_n = min(start + args.batch_size, len(pool))
        print(f"[mine] {done_n}/{len(pool)} prompts this session | "
              f"pairs total: {n_pairs_total} | 0-pass bucket: {n_sft_total}")

    # ---- recompute everything from the full ledger (all sessions) ----
    ledger = load_ledger(args.out_dir)
    histogram = collections.Counter(r["n_pass"] for r in ledger)
    ambig_total = sum(r.get("n_ambiguous", 0) for r in ledger)
    pairs = [json.loads(l) for l in open(pairs_path)] \
        if os.path.exists(pairs_path) else []

    # malformed-share cap (guardrail: <=5% of pairs), applied to the file
    mal = [p for p in pairs if p["meta"]["error_type"] == "malformed_syntax"]
    cap = max(1, int(0.05 * len(pairs))) if pairs else 0
    if len(mal) > cap:
        drop_ids = {p["meta"]["pair_id"] for p in mal[cap:]}
        pairs = [p for p in pairs if p["meta"]["pair_id"] not in drop_ids]
        with open(pairs_path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"[mine] malformed cap: dropped {len(drop_ids)} pair(s)")

    sft_count = sum(1 for r in ledger if r["outcome"] == "sft_bucket")
    for name, count in (("mined_pairs.jsonl", len(pairs)),
                        ("sft_bucket.jsonl", sft_count),
                        ("ambiguous_review.jsonl", ambig_total)):
        print(f"[mine] {count:5d} row(s) in {os.path.join(args.out_dir, name)}")

    # --------------------------------------------------------- decision ----
    n = len(ledger)
    print("\n=== pass-rate histogram (prompts by #correct of "
          f"{args.samples} samples) ===")
    for k in range(args.samples + 1):
        bar = "#" * int(60 * histogram[k] / max(1, n))
        print(f"  {k}/{args.samples}: {histogram[k]:5d} {bar}")
    print(f"  ambiguous generations excluded from both sides: {ambig_total}")
    yield_rate = len(pairs) / max(1, n)
    summary = {
        "prompts_mined": n, "pairs": len(pairs),
        "sft_bucket": sft_count, "yield_rate": round(yield_rate, 4),
        "projected_pairs_at_5k_prompts": int(yield_rate * 5000),
        "projected_pairs_at_10k_prompts": int(yield_rate * 10000),
        "histogram": {f"{k}/{args.samples}": histogram[k]
                      for k in range(args.samples + 1)},
    }
    proj = summary["projected_pairs_at_10k_prompts"]
    if proj >= 1000:
        decision = ("PROCEED with the DPO rerun: projected >=1000 hard "
                    "on-policy pairs at a 10k-prompt sweep.")
    elif proj >= 300:
        decision = ("PROCEED CAUTIOUSLY: projected 300-1000 pairs. 1 epoch "
                    "max, eval callback every ~50 steps.")
    else:
        decision = ("DO NOT RUN DPO on this yield. The inconsistency zone is "
                    "too thin; rejection-sampling SFT on sft_bucket.jsonl is "
                    "the pre-registered alternative, and this result is a "
                    "publishable finding in its own right.")
    summary["pre_registered_decision"] = decision
    with open(os.path.join(args.out_dir, "mining_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== pre-registered decision ===")
    print(decision)
    print(f"\n[mine] summary -> {os.path.join(args.out_dir, 'mining_summary.json')}")
    print("[mine] reminder: manually read 50 random rows of mined_pairs.jsonl "
          "and every row of ambiguous_review.jsonl before training "
          "(verifier-audit guardrail).")


# --------------------------------------------------------------------- cli -

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run verifier unit tests against the fixture set and exit")
    ap.add_argument("--skip-self-test", action="store_true",
                    help="mine without the fixture self-test gate (not recommended)")
    ap.add_argument("--fixtures", default="./fixtures",
                    help="directory containing fixture_pairs_*.jsonl")
    ap.add_argument("--adapter",
                    help="path or HF id of the shipped SFT LoRA adapter (required for mining)")
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dataset", default="NousResearch/hermes-function-calling-v1")
    ap.add_argument("--dataset-config", default="func_calling_singleturn")
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="prompts per generation batch (x samples sequences)")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--bfcl-path", nargs="*", default=None,
                    help="BFCL jsonl file(s) for n-gram + schema decontamination")
    ap.add_argument("--out-dir", default="mining_out")
    ap.add_argument("--fresh", action="store_true",
                    help="clear any previous ledger/outputs in --out-dir and start over")
    ap.add_argument("--redo-last", type=int, metavar="N", default=None,
                    help="roll back the last N processed prompts in --out-dir so "
                         "they get re-mined; without --adapter this rolls back and exits")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test(args.fixtures) else 1)
    if args.fresh:
        for name in ("ledger.jsonl", "mined_pairs.jsonl", "sft_bucket.jsonl",
                     "ambiguous_review.jsonl", "mining_summary.json"):
            p = os.path.join(args.out_dir, name)
            if os.path.exists(p):
                os.remove(p)
        print(f"[mine] --fresh: cleared previous outputs in {args.out_dir}")
    if args.redo_last is not None:
        redo_last(args.out_dir, args.redo_last)
        if not args.adapter:
            sys.exit(0)
    if not args.adapter:
        ap.error("--adapter is required for mining (or use --self-test / --redo-last)")
    if not args.skip_self_test:
        print("[mine] running fixture self-test gate first...")
        if not self_test(args.fixtures):
            print("[mine] self-test FAILED; refusing to mine. Fix the "
                  "verifier or pass --skip-self-test to override.",
                  file=sys.stderr)
            sys.exit(1)
    run_mining(args)


if __name__ == "__main__":
    main()
