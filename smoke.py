"""Week 1 smoke test.

Verifies that the local Python environment can load a small Llama variant
and run one inference. Uses Llama-3.2-1B-Instruct (not the target 8B model)
because 1B fits comfortably in laptop memory. The 8B model will run on
rented cloud GPUs starting Week 2.

Prerequisites (see docs/learning/week-1-fundamentals.md):
- HuggingFace account with API token exported as HF_TOKEN
- Accepted the Llama-3.2 license on huggingface.co
- Python 3.11+, transformers, torch, huggingface_hub installed
- Run `huggingface-cli login` once (uses HF_TOKEN)

Usage:
    python smoke.py

Success: the script prints a one-sentence explanation of a transformer.
First run downloads the model (~2.5 GB) to ~/.cache/huggingface/.
Subsequent runs use the cache. CPU generation takes 30-60 seconds.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def main() -> None:
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
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
    )

    print("Generating...")
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][inputs.shape[1]:], skip_special_tokens=True
    )
    print("\n--- Output ---")
    print(response)


if __name__ == "__main__":
    main()
