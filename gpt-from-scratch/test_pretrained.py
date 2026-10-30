"""Test that a pretrained GPT-2 model + tokenizer work with this architecture."""

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from gpt3 import GPT3


def test_pretrained_load_and_generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and pretrained model into the custom architecture.
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = GPT3.from_pretrained("openai-community/gpt2").to(device)
    model.eval()

    # Sanity check: custom model logits should match HF GPT-2 logits.
    hf_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)
    hf_model.eval()

    prompt = "The quick brown fox"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        custom_out = model(input_ids)
        hf_out = hf_model(input_ids)

    logit_diff = (custom_out["logits"] - hf_out.logits).abs().max().item()
    print(f"Max logit diff vs HF GPT-2: {logit_diff:.6f}")
    assert logit_diff < 1e-4, f"Logits mismatch: {logit_diff}"

    # Generate text greedily.
    generated = model.generate(input_ids, max_new_tokens=20)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated: {text!r}")

    # Decode should start with the prompt.
    assert text.startswith(prompt), f"Generation did not start with prompt: {text!r}"
    print("Pretrained model + tokenizer test passed.")


if __name__ == "__main__":
    test_pretrained_load_and_generate()
