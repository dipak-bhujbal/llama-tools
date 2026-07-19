"""Publish the preference dataset to HuggingFace Hub.

Uploads `data/processed/preferences.jsonl` as a HF dataset repo with a
proper dataset card. The dataset is one of the project's permanent
artifacts (see ARCHITECTURE.md "Publishing artifacts").

Starts PRIVATE — flipping to public is part of the Week 10-12 release
polish, alongside the model cards. One command flips it later:
    hf repo update centuriandip/tool-calling-preferences --private false

Usage
-----
    python data/push_to_hf.py

Input:  data/processed/preferences.jsonl
Output: https://huggingface.co/datasets/centuriandip/tool-calling-preferences
"""

import json
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

INPUT_PATH = Path("data/processed/preferences.jsonl")
HF_REPO_ID = "centuriandip/tool-calling-preferences"
PRIVATE = True

DATASET_CARD = """\
---
license: cc-by-4.0
language:
- en
task_categories:
- text-generation
tags:
- function-calling
- tool-use
- dpo
- preference-data
size_categories:
- 10K<n<100K
---

# tool-calling-preferences

Preference pairs for DPO training of tool-calling models. Each example
contains a prompt (system message with tool schemas + user query), a
`chosen` response (correct tool call), and a `rejected` response (an
adversarially perturbed variant).

## Construction

- **Source:** derived from [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
  (CC-BY-4.0), deduplicated with MinHash+LSH (Jaccard 0.7, 128 perms,
  word-level 5-grams).
- **Rejected generation:** rule-based adversarial perturbation of the
  correct tool call. One of five perturbation types per pair:

| Perturbation | Share | Description |
|---|---|---|
| `malformed_json` | ~22% | JSON corrupted (trailing comma, unterminated string, missing brace) |
| `wrong_arg_value` | ~22% | One argument value replaced with a plausibly-wrong value |
| `missing_required_arg` | ~21% | One provided argument removed |
| `hallucinated_tool` | ~21% | Tool name replaced with a plausible but nonexistent tool |
| `wrong_tool_from_list` | ~14% | Tool name replaced with a different valid tool from the same list |

- **Ratio:** 1:1 (one rejected per chosen).
- **Deterministic:** fixed seed; reproducible from the
  [llama-tools](https://github.com/dipak-bhujbal/llama-tools) pipeline
  (`data/assemble_sft.py` → `data/dedupe.py` → `data/assemble_preferences.py`).

## Fields

- `prompt_messages` — list of `{role, content}` for system + user turns
- `chosen` — correct assistant tool call (JSON string)
- `rejected` — perturbed assistant tool call (JSON string)
- `perturbation_type` — which of the five perturbations was applied
- `source` / `source_id` — provenance back to the origin dataset

## Intended use

DPO / preference-optimization training of open-weight models for
structured tool-calling. Built as the preference stage for
[llama-3.1-8b-tools](https://huggingface.co/centuriandip/llama-3.1-8b-tools).

## Limitations

- Rejected examples are rule-generated, not sampled from real model
  failures. The taxonomy is documented and deterministic, but does not
  cover failure modes outside the five perturbation types.
- English-only. Single- and multi-call examples; the perturbation is
  applied to the first call in multi-call examples.

## License

CC-BY-4.0 (inherited from the xLAM source data). Attribution:
Salesforce xLAM team for the underlying corpus.
"""


def main() -> None:
    load_dotenv()

    if not INPUT_PATH.exists():
        raise SystemExit(
            f"Input not found: {INPUT_PATH}\n"
            f"Run `python data/assemble_preferences.py` first."
        )

    rows = []
    with open(INPUT_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} preference pairs")

    ds = Dataset.from_list(rows)
    print(f"Dataset: {ds}")

    print(f"Pushing to {HF_REPO_ID} (private={PRIVATE})...")
    ds.push_to_hub(HF_REPO_ID, private=PRIVATE)

    # Upload the dataset card. push_to_hub auto-generates a minimal card;
    # we overwrite with the full one.
    api = HfApi()
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )

    print(f"\nDone: https://huggingface.co/datasets/{HF_REPO_ID}")
    print("Dataset card uploaded. Currently PRIVATE — flip to public at release:")
    print(f"  https://huggingface.co/datasets/{HF_REPO_ID}/settings")


if __name__ == "__main__":
    main()
