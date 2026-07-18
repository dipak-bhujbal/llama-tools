# model_card/

Model card source materials. Populated in **Week 10** (final release polish).

## What lands here

- **`llama-3.1-8b-tools.md`** — the model card for the full-precision release, following the [HuggingFace model card guidelines](https://huggingface.co/docs/hub/model-cards). Pushed to `huggingface.co/centuriandip/llama-3.1-8b-tools`.
- **`llama-3.1-8b-tools-awq.md`** — model card for the AWQ int4 variant.
- **`examples/`** — before/after prompt-and-response pairs used in the model card to illustrate the improvement over base Llama-3.1-8B-Instruct.

## Required sections (per HF template)

- Model details (base, license, training procedure)
- Intended use + out-of-scope use
- Bias, risks, and limitations
- Training data (with dataset link)
- Training procedure (SFT + DPO configs, hyperparameters)
- Evaluation (BFCL v3 results + MMLU regression)
- Environmental impact (compute-hours, provider)
- Citation

## Not in v1

- Multilingual documentation (English only)
- Interactive demo widget (defer to v2)
