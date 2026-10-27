import torch

from gpt3 import GPT3, GPT3Config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    "checkpoints/best.pt",
    map_location=device,
    weights_only=True,
)

config = GPT3Config(**checkpoint["config"])

model = GPT3(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

torch.manual_seed(42)

input_ids = torch.randint(
    0,
    config.vocab_size,
    (1, 8),
    device=device,
)

with torch.no_grad():
    # Reference: process all tokens in one pass.
    full_logits = model(input_ids)["logits"]

    # Process the first token and initialize the cache.
    output = model(input_ids[:, :1], use_cache=True)
    cache = output["past_key_values"]

    max_error = (
        output["logits"][:, -1, :] - full_logits[:, 0, :]
    ).abs().max().item()

    # Feed subsequent tokens one at a time.
    for pos in range(1, input_ids.size(1)):
        output = model(
            input_ids[:, pos:pos + 1],
            past_key_values=cache,
            use_cache=True,
        )

        cache = output["past_key_values"]

        error = (
            output["logits"][:, -1, :] - full_logits[:, pos, :]
        ).abs().max().item()

        max_error = max(max_error, error)

print(f"Maximum logit difference: {max_error:.8f}")

assert max_error < 1e-4, "Cached and uncached outputs disagree!"

print("KV cache correctness test passed.")