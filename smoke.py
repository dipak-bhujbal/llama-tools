"""Week 1 smoke test.

Verifies that the local Python environment can load a small model and run
one inference. Uses SmolLM2-135M-Instruct — a tiny (~135M-parameter),
non-gated model that runs comfortably on CPU and downloads in seconds.

The purpose is to verify that transformers + torch + huggingface_hub are
correctly installed and can pull a model from the Hub. It is deliberately
NOT Llama: the smoke test should not depend on a gated-model approval.
Our actual target model (Llama-3.1-8B-Instruct) is verified separately at
environment-setup time by checking config-file access; it runs on cloud
GPUs starting Week 2, not on your laptop.

Prerequisites (see docs/learning/week-1-fundamentals.md):
- HuggingFace account with API token exported as HF_TOKEN
- Python 3.11+, transformers, torch, huggingface_hub installed
- Optionally: `hf auth login --token $HF_TOKEN` for persistent auth

Usage:
    python smoke.py

Success: the script prints a one-sentence explanation of a transformer.
First run downloads the model (~300 MB) to ~/.cache/huggingface/.
Subsequent runs use the cache. CPU generation takes 5-15 seconds.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def main() -> None:
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading model {MODEL_ID}...")
    # Load in fp32 on CPU. No `device_map` — that would require the
    # `accelerate` package, which is unnecessary overhead for a smoke test.
    # On Mac, the default is CPU; on Linux CUDA, the model would still load
    # to CPU without explicit .to(device) — fine for a one-off inference.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
    )

    messages = [
        {"role": "system", "content": "You are a concise, helpful assistant."},
        {
            "role": "user",
            "content": "In one sentence, what is a transformer in machine learning?",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    print("Generating...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print("\n--- Output ---")
    print(response)


if __name__ == "__main__":
    main()
