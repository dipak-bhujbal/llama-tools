# data/

Data curation pipeline. Populated in **Week 3**.

## What lands here

Scripts that assemble, deduplicate, format, and validate the training datasets:

- **`assemble_sft.py`** — pulls Glaive function-calling + APIGen + xLAM-function-calling-60k, normalizes to a unified schema, writes to `data/processed/sft.jsonl`
- **`assemble_preferences.py`** — builds preference pairs: chosen = correct tool call, rejected = adversarially perturbed variant (wrong function, missing arg, malformed JSON, hallucinated tool). Writes `data/processed/preferences.jsonl`
- **`dedupe.py`** — MinHash+LSH deduplication against BFCL v3 eval set to prevent leakage
- **`format_chat.py`** — applies Llama-3.1 chat template to all examples
- **`push_to_hf.py`** — publishes the final preference dataset as `centuriandip/tool-calling-preferences`
- **`validate.py`** — jsonschema validation of tool-call JSON against function schemas

## Directory layout (after Week 3)

```
data/
├── README.md              # this file
├── assemble_sft.py
├── assemble_preferences.py
├── dedupe.py
├── format_chat.py
├── push_to_hf.py
├── validate.py
├── raw/                   # git-ignored — downloaded source datasets
├── processed/             # git-ignored — cleaned outputs (large)
└── manifests/             # committed — provenance + version metadata for each dataset
```

## Not in v1

- Human-annotated preference pairs (v1 uses adversarial synthesis only)
- Multi-language function calling (English only)
- Multimodal (text-only tool calls)
