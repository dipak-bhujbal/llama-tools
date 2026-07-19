"""Assemble the SFT training set from Hermes + xLAM.

Purpose
-------
Pull two public function-calling datasets from HuggingFace, normalize both
into a single common "messages" schema, and write a combined JSONL file
suitable for the Week 4 SFT training run.

Sources (see docs/decisions/ADR-003-source-datasets.md):
- `NousResearch/hermes-function-calling-v1` (config: `func_calling`) — already
  in ShareGPT-style messages format, just needs field renaming.
- `Salesforce/xlam-function-calling-60k` — has structured (query, tools,
  answers) fields; needs conversion to messages format.

Output format
-------------
Each line is a JSON object with the shape:

    {
      "messages": [
        {"role": "system", "content": "You are a helpful AI assistant.\\n\\nTools available:\\n<serialized tool schemas>"},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "<serialized tool call OR natural language>"}
      ],
      "source": "hermes" | "xlam",
      "source_id": "<original example ID>"
    }

Design notes
------------
- We keep `messages` in OpenAI-style format because TRL's SFTTrainer applies
  the Llama-3.1 chat template automatically when the training column is
  called "messages" and each turn is a dict with role + content.
- Tool schemas are serialized as JSON inside the system prompt. This is the
  simplest approach that works cross-model and matches how most fine-tuned
  tool-calling models format their training data.
- Assistant tool calls are serialized as JSON inside the content field, not
  as separate `tool_calls` list — again, simplest for SFT.
- Provenance (`source`, `source_id`) is preserved per ADR-003 so we can trace
  quality issues back to a source dataset later.

This script is deliberately single-file with no CLI arguments. Configuration
is at the top; edit values there, not via env vars, so runs are reproducible
from a git commit.

Usage
-----
    python data/assemble_sft.py

Output: `data/processed/sft.jsonl` (one JSON object per line).
"""

import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

HERMES_REPO = "NousResearch/hermes-function-calling-v1"
# Hermes ships several configs. `func_calling` is multi-turn (~1.9K examples);
# `func_calling_singleturn` is single-turn (~5.7K). Together they give ~7.6K
# with a healthy mix of conversation shapes. Skip json_mode configs — those
# target structured JSON output, not tool-use.
HERMES_CONFIGS = ["func_calling", "func_calling_singleturn"]
HERMES_TARGET = 7500

XLAM_REPO = "Salesforce/xlam-function-calling-60k"
XLAM_TARGET = 7500  # requires access approval from Salesforce; script degrades
                    # gracefully if not yet approved (Hermes-only mode)

OUTPUT_PATH = Path("data/processed/sft.jsonl")
SEED = 42


# --------------------------------------------------------------------------
# System-prompt template
# --------------------------------------------------------------------------

def build_system_prompt(tools: list[dict[str, Any]]) -> str:
    """Serialize tool schemas into a Llama-friendly system prompt.

    We put the tools as pretty-printed JSON inside a natural-language wrapper.
    The Llama-3.1 chat template will handle wrapping this in the appropriate
    special tokens; we just need the abstract content string.
    """
    if not tools:
        return "You are a helpful AI assistant."
    tools_json = json.dumps(tools, indent=2)
    return (
        "You are a helpful AI assistant. You have access to the following tools. "
        "When you need to use a tool, respond with a JSON object of the form "
        '{"name": "<tool_name>", "arguments": {...}}.\n\n'
        f"Tools:\n{tools_json}"
    )


# --------------------------------------------------------------------------
# Hermes loader + normalizer
# --------------------------------------------------------------------------

def load_hermes() -> list[dict[str, Any]]:
    """Load Hermes function-calling (all tool-use configs) and normalize.

    Hermes configs have a `conversations` field: list of ShareGPT-style
    turns (`{"from": role, "value": content}`). We combine `func_calling`
    (multi-turn) and `func_calling_singleturn` for enough volume.
    """
    normalized: list[dict[str, Any]] = []
    role_map = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
    total_loaded = 0
    for config in HERMES_CONFIGS:
        print(f"Loading {HERMES_REPO} ({config})...")
        raw = load_dataset(HERMES_REPO, config, split="train")
        total_loaded += len(raw)
        for i, row in enumerate(raw):
            turns = row.get("conversations", [])
            if not turns:
                continue
            messages = []
            for t in turns:
                role = role_map.get(t.get("from", ""))
                content = t.get("value", "")
                if role and content:
                    messages.append({"role": role, "content": content})
            if len(messages) < 3:
                continue
            normalized.append(
                {
                    "messages": messages,
                    "source": "hermes",
                    "source_id": f"hermes-{config}-{i}",
                }
            )
    print(f"  Hermes: loaded {total_loaded}, kept {len(normalized)}")
    return normalized


# --------------------------------------------------------------------------
# xLAM loader + normalizer
# --------------------------------------------------------------------------

def load_xlam() -> list[dict[str, Any]]:
    """Load xLAM function-calling and normalize to the common schema.

    xLAM structure per example:
        query: str                          — user request
        tools: str (JSON list of tool defs) — available tools
        answers: str (JSON list of tool calls) — expected assistant response

    xLAM is gated on HuggingFace. If access is not yet granted, this function
    returns an empty list and prints a friendly message; the caller falls
    back to Hermes-only mode.
    """
    print(f"Loading {XLAM_REPO}...")
    try:
        raw = load_dataset(XLAM_REPO, split="train")
    except Exception as e:
        msg = str(e)
        if "gated" in msg or "access" in msg or "authorized" in msg:
            print(
                "  xLAM: SKIPPED — you don't have access yet.\n"
                "  Request access at: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k\n"
                "  Then rerun this script. Hermes-only mode continues below."
            )
        else:
            print(f"  xLAM: SKIPPED due to error: {type(e).__name__}: {msg[:200]}")
        return []

    normalized: list[dict[str, Any]] = []
    for i, row in enumerate(raw):
        query = row.get("query", "").strip()
        tools_raw = row.get("tools", "[]")
        answers_raw = row.get("answers", "[]")
        if not query:
            continue
        try:
            tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
            answers = json.loads(answers_raw) if isinstance(answers_raw, str) else answers_raw
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(tools, list) or not isinstance(answers, list) or not answers:
            continue

        # Build 3-turn conversation: system (with tools) → user (query) → assistant (calls).
        system_prompt = build_system_prompt(tools)
        assistant_content = json.dumps(answers) if len(answers) > 1 else json.dumps(answers[0])
        normalized.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": assistant_content},
                ],
                "source": "xlam",
                "source_id": f"xlam-{i}",
            }
        )
    print(f"  xLAM: loaded {len(raw)}, kept {len(normalized)}")
    return normalized


# --------------------------------------------------------------------------
# Combine + sample + write
# --------------------------------------------------------------------------

def main() -> None:
    random.seed(SEED)

    hermes = load_hermes()
    xlam = load_xlam()

    # Sample down to target counts. Random-sampling preserves source diversity
    # rather than always taking the first N (which could be systematically
    # biased if the dataset is ordered by creation date, topic, etc.).
    hermes_sample = random.sample(hermes, min(HERMES_TARGET, len(hermes)))
    xlam_sample = random.sample(xlam, min(XLAM_TARGET, len(xlam)))

    combined = hermes_sample + xlam_sample
    random.shuffle(combined)  # interleave sources so a downstream reader
                              # sees mixed provenance in training order

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for ex in combined:
            f.write(json.dumps(ex) + "\n")

    print("\nWrote:", OUTPUT_PATH)
    print(f"  Total examples: {len(combined)}")
    print(f"  From Hermes:    {len(hermes_sample)}")
    print(f"  From xLAM:      {len(xlam_sample)}")
    if not xlam_sample:
        print(
            "\n  NOTE: xLAM was skipped — SFT set is Hermes-only for now.\n"
            "  Once you have xLAM access, rerun to get the full 15K set."
        )
    print("\nNext step: `python data/dedupe.py` to remove BFCL overlap.")


if __name__ == "__main__":
    main()
