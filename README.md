# llama-tools

Post-trained Llama-3.1-8B for structured tool-calling in AI agents.

## What this is

An open-weight fine-tune of `meta-llama/Llama-3.1-8B-Instruct` optimized for reliable function-calling behavior: correct tool selection, correct argument formatting, correct types, no hallucinated tools. Full SFT + DPO training pipeline, evaluated on the Berkeley Function Calling Leaderboard (BFCL v3), with quantized variants for production serving.

**Status:** In development. Not yet released. See [`PLAN.md`](./PLAN.md) for the week-by-week timeline.

## Quick links (placeholders — not yet live)

- Model: `huggingface.co/centuriandip/llama-3.1-8b-tools` *(release target)*
- Preference dataset: `huggingface.co/datasets/centuriandip/tool-calling-preferences` *(release target)*
- Technical report: [`docs/technical-report.md`](./docs/technical-report.md) *(pending)*

## What's in this repo

```
llama-tools/
├── PLAN.md                        # Week-by-week execution plan
├── ARCHITECTURE.md                # Technical design
├── data/                          # Data curation scripts (assembly, dedup, filtering)
├── train/                         # SFT and DPO training scripts (TRL-based)
├── eval/                          # BFCL v3 harness + MMLU regression check
├── quantize/                      # AWQ int4 quantization pipeline
├── model_card/                    # Model card source materials
└── docs/
    ├── decisions/                 # Architecture Decision Records (ADRs)
    ├── learning/                  # Week-by-week learning ramp
    └── technical-report.md        # Publishable writeup (v1)
```

## Reproducing

Full reproduction instructions land with v1 release. The repo is under active development; the training scripts, data pipeline, and eval harness will stabilize week-by-week per [`PLAN.md`](./PLAN.md).

## Related projects

- **[release-kit](https://github.com/dipak-bhujbal/release-kit)** — open framework for eval-gated LLM releases. `llama-tools` is its reference implementation.

## License

Apache-2.0

## Author

[Dipak Bhujbal](https://github.com/dipak-bhujbal)
