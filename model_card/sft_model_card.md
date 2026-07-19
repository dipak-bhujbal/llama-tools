---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B-Instruct
language:
- en
tags:
- tool-calling
- function-calling
- lora
- sft
pipeline_tag: text-generation
---

# llama-3.1-8b-tools-sft

SFT-only checkpoint of Llama-3.1-8B-Instruct fine-tuned for structured tool and
function calling. This is **stage one** of a two-stage pipeline (SFT → DPO). A
preference-optimized variant built on top of this checkpoint is coming in a later
week (see the [llama-tools](https://github.com/dipak-bhujbal/llama-tools) build
plan).

## Model summary

| Field | Value |
|---|---|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| Training method | LoRA-SFT (adapter merged into base weights) |
| Training set | 12,160 deduped examples (Hermes + xLAM) |
| Eval set | 500 held-out examples (random split, seed 42) |
| Hardware | 1× RTX A6000 48 GB (Runpod) |
| Wall-clock time | ~9 hrs |

## Intended use

**Primary use case:** structured tool and function calling with open-weight models.
Given a system prompt containing serialized tool schemas and a user query, the model
responds with a JSON tool call of the form `{"name": "<tool_name>", "arguments": {...}}`.

**Out-of-scope uses:** general-purpose chat (the fine-tune shifts the distribution
toward tool-use responses), multilingual queries, multimodal inputs, safety-critical
applications without additional red-teaming.

## Training data

### Sources

| Source | Loaded | After dedup | Share of final set |
|---|---|---|---|
| `NousResearch/hermes-function-calling-v1` (`func_calling` + `func_calling_singleturn`) | 3,786 | 1,161 | ~10% |
| `Salesforce/xlam-function-calling-60k` | 11,214 sampled from 60K | 10,999 | ~90% |
| **Total** | **15,000** | **12,160** | |

See [ADR-003](https://github.com/dipak-bhujbal/llama-tools/blob/main/docs/decisions/ADR-003-source-datasets.md)
for the dataset selection rationale.

### Deduplication

Near-duplicate removal via MinHash + LSH on the user query field. Configuration:
128 permutations, Jaccard similarity threshold 0.7, word-level 5-grams.
The 15,000-example pre-dedup set dropped 2,840 examples (18.9%), yielding 12,160
for training. The high Hermes drop rate (69%) reflects templated queries in the
`func_calling_singleturn` config — the post-dedup signal is dominated by xLAM's more
varied query distribution.

**BFCL v3 leakage dedup:** deferred to Week 7 when the eval harness lands. The BFCL
v3 HF data schema shifts as new task categories are added; wiring dedup against it
prematurely would give a false sense of safety. This is a known limitation of this
checkpoint — see the Limitations section.

### Chat template and format

All examples are formatted with the Llama-3.1 chat template via TRL's `SFTTrainer`.
The dataset column is `messages` (list of `{role, content}` dicts); TRL applies the
template automatically. Tool schemas are serialized as JSON inside the system prompt:

```
You are a helpful AI assistant. You have access to the following tools.
When you need to use a tool, respond with a JSON object of the form
{"name": "<tool_name>", "arguments": {...}}.

Tools:
[<pretty-printed JSON array of tool definitions>]
```

Assistant tool calls are serialized as JSON in the content field, not as a separate
`tool_calls` list. Multi-call responses are JSON arrays.

A companion preference dataset with 10,999 DPO pairs is published separately at
[centuriandip/tool-calling-preferences](https://huggingface.co/datasets/centuriandip/tool-calling-preferences).

## Training procedure

### Hyperparameters

| Parameter | Value |
|---|---|
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Epochs | 3 |
| Per-device batch size | 8 |
| Gradient accumulation steps | 4 |
| Effective batch size | 32 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 3% |
| Precision | bf16 |
| Max sequence length | 2048 |
| Gradient checkpointing | enabled |
| Packing | disabled |

Training split: 11,660 examples train / 500 held-out eval (random split, seed 42).
Approximate optimizer steps: ~1,090.

### Hardware and environment

- GPU: 1× RTX A6000 48 GB (Runpod spot)
- Wall-clock: approximately 9 hours
- Frameworks: `transformers`, `trl`, `peft`, `torch` (bf16)

The adapter was trained with PEFT/LoRA and then merged into the base weights via
`merge_and_unload()` before upload. The uploaded model is a full-weight merge, not
an adapter-only repo.

### Training curves

<!-- TODO: add final train loss and eval loss values once the run completes tonight -->
<!-- TODO: link wandb run — project: llama-tools, run: sft-full-week4 -->

| Metric | Value |
|---|---|
| Final train loss | TODO |
| Best eval loss | TODO |
| wandb report | TODO |

## Evaluation

### Held-out eval loss

500 examples were held out from the training set and used for in-training eval loss
monitoring every 100 optimizer steps. Loss curves are available in the linked wandb
report (TODO above).

### BFCL v3 (planned)

Formal evaluation against the Berkeley Function-Calling Leaderboard v3 is scheduled
for Week 7. This checkpoint, the base `meta-llama/Llama-3.1-8B-Instruct`, and the
later SFT+DPO checkpoint will be scored across all four BFCL v3 categories
(simple, multiple, parallel, nested), with MMLU as a general-capability regression
check. Results will be committed to `eval/results/week-7.md` and linked here.

<!-- TODO: replace this section with the BFCL v3 comparison table after Week 7 -->

## How to use

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="centuriandip/llama-3.1-8b-tools-sft",
    torch_dtype="bfloat16",
    device_map="auto",
)

# Tool schema(s) to pass in the system prompt.
import json

tools = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "location": {
                "description": "City and country, e.g. 'San Francisco, US'.",
                "type": "str",
            },
            "unit": {
                "description": "Temperature unit: 'celsius' or 'fahrenheit'.",
                "type": "str",
                "default": "celsius",
            },
        },
    }
]

system_prompt = (
    "You are a helpful AI assistant. You have access to the following tools. "
    "When you need to use a tool, respond with a JSON object of the form "
    '{"name": "<tool_name>", "arguments": {...}}.\n\n'
    f"Tools:\n{json.dumps(tools, indent=2)}"
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's the weather like in Tokyo right now?"},
]

output = pipe(messages, max_new_tokens=256, do_sample=False)
print(output[0]["generated_text"][-1]["content"])
# Expected output format:
# {"name": "get_current_weather", "arguments": {"location": "Tokyo, JP", "unit": "celsius"}}
```

For multi-call responses the model outputs a JSON array:
```json
[
  {"name": "tool_a", "arguments": {"arg1": "value1"}},
  {"name": "tool_b", "arguments": {"arg2": "value2"}}
]
```

Parse the assistant content with `json.loads()` and dispatch accordingly. No special
tokenizer tool-call tokens are used — everything is in the content string.

## Limitations

- **SFT-only.** This checkpoint has not undergone preference optimization. It learns
  to reproduce correct tool calls but has not been explicitly trained to avoid
  near-miss failures (wrong tool selected, malformed arguments, hallucinated tool
  names). The DPO variant addresses this; results will be published as
  `centuriandip/llama-3.1-8b-tools` once Week 6 completes.

- **No BFCL v3 leakage dedup.** Training data has not been deduplicated against the
  BFCL v3 eval set. Formal evaluation should be treated with that caveat in mind.
  Leakage dedup is scheduled for Week 7 and will be applied before final model
  reporting.

- **English-centric.** Both Hermes and xLAM are predominantly English. Tool-calling
  behavior in other languages is untested and likely degraded.

- **Single-turn and simple multi-turn only.** The xLAM source is single-turn; Hermes
  includes some multi-turn conversations. The model has not been trained on long
  agentic chains or feedback loops.

- **Inherits base-model limitations.** This model inherits all limitations of
  `meta-llama/Llama-3.1-8B-Instruct`, including knowledge cutoff, bias from
  pretraining data, and susceptibility to prompt injection. See Meta's model card
  for the full list.

- **Rule-based rejection taxonomy.** The companion preference dataset uses
  adversarial perturbations, not real model failures. Coverage is limited to five
  perturbation types; novel failure modes outside that taxonomy are not represented
  in preference training.

## Sample outputs

<!-- TODO: add 3-5 before/after examples (base vs this checkpoint) after the run
     completes tonight. Pull from outputs/sft-full/sample_generations.json. -->

## Environmental impact

- Hardware: 1× RTX A6000 48 GB
- Provider: Runpod (cloud spot)
- Training duration: ~9 hours
- Estimated CO₂ impact: not yet calculated

## Citation

```
@misc{bhujbal2026llamatools,
  author    = {Dipak Bhujbal},
  title     = {llama-3.1-8b-tools-sft: LoRA-SFT of Llama-3.1-8B-Instruct for Structured Tool Calling},
  year      = {2026},
  url       = {https://huggingface.co/centuriandip/llama-3.1-8b-tools-sft},
  note      = {SFT-only checkpoint; DPO variant forthcoming as centuriandip/llama-3.1-8b-tools}
}
```

## License

This model is released under the
[Meta Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/).
Use of this model requires accepting Meta's license terms. Downstream use must
include appropriate attribution.

Training data licenses: Hermes (Apache-2.0), xLAM (CC-BY-4.0).
