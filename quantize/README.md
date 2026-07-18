# quantize/

Quantization pipeline. Populated in **Week 9**.

## What lands here

- **`awq_quantize.py`** — AWQ int4 quantization via [AutoAWQ](https://github.com/casper-hansen/AutoAWQ). Uses a held-out calibration set drawn from tool-calling examples.
- **`benchmark_quantized.py`** — re-runs BFCL v3 on the quantized model. Reports quality delta vs full-precision (target: <2% degradation) and memory reduction (target: 3-4x).
- **`push_awq.py`** — publishes the quantized model to HF as `centuriandip/llama-3.1-8b-tools-awq`.

## Not in v1

- GPTQ int4 variant (defer if AWQ quality is acceptable)
- int8 / fp8 variants
- SmoothQuant, ZeroQuant, or other advanced quantization methods
- Quantization-aware fine-tuning

## Related

- `../train/` — full-precision model produced here first
- `../eval/` — used to measure post-quantization quality
