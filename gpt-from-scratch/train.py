# train.py

import os
import math
import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from gpt3 import GPT3, GPT3Config


# -----------------------
# Configuration
# -----------------------

SEED = 42
BATCH_SIZE = 2
SEQ_LEN = 128
TRAIN_STEPS = 1000
EVAL_INTERVAL = 100
EVAL_BATCHES = 20
LEARNING_RATE = 3e-4

os.makedirs("checkpoints", exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")


# -----------------------
# Load dataset + tokenizer
# -----------------------

print("Loading WikiText-2...")

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
)

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

def tokenize_split(split):
    # Preserve line boundaries instead of merging words across lines.
    text = "\n".join(dataset[split]["text"])
    ids = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long)

train_tokens = tokenize_split("train")
val_tokens = tokenize_split("validation")

print(f"Training tokens:   {len(train_tokens):,}")
print(f"Validation tokens: {len(val_tokens):,}")


# -----------------------
# Batch generation
# -----------------------

def get_batch(tokens):
    # Each sampled sequence has SEQ_LEN input tokens.
    starts = torch.randint(
        0,
        len(tokens) - SEQ_LEN - 1,
        (BATCH_SIZE,),
    )

    x = torch.stack([
        tokens[s : s + SEQ_LEN]
        for s in starts.tolist()
    ])

    return x.to(device)


# -----------------------
# Model
# -----------------------

config = GPT3Config(
    vocab_size=len(tokenizer),
    max_seq_len=SEQ_LEN,
    d_model=192,
    n_heads=6,
    n_layers=4,
    dropout=0.1,
)

model = GPT3(config).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.1,
)


# -----------------------
# Evaluation
# -----------------------

@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for split, tokens in [
        ("train", train_tokens),
        ("validation", val_tokens),
    ]:
        losses = []

        for _ in range(EVAL_BATCHES):
            x = get_batch(tokens)
            output = model(x, labels=x)
            losses.append(output["loss"].item())

        results[split] = sum(losses) / len(losses)

    model.train()
    return results


# -----------------------
# Training loop
# -----------------------

best_val_loss = float("inf")

for step in range(1, TRAIN_STEPS + 1):
    model.train()

    x = get_batch(train_tokens)

    output = model(x, labels=x)
    loss = output["loss"]

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Prevent occasional large gradient updates.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    if step % EVAL_INTERVAL == 0 or step == 1:
        losses = estimate_loss()

        # Perplexity is exp(cross-entropy loss).
        train_ppl = math.exp(min(losses["train"], 20))
        val_ppl = math.exp(min(losses["validation"], 20))

        print(
            f"Step {step:4d} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['validation']:.4f} | "
            f"train ppl {train_ppl:.2f} | "
            f"val ppl {val_ppl:.2f}"
        )

        if losses["validation"] < best_val_loss:
            best_val_loss = losses["validation"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(config),
                    "step": step,
                    "val_loss": best_val_loss,
                },
                "checkpoints/best.pt",
            )

print("Training complete.")
print("Best checkpoint: checkpoints/best.pt")