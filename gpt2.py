"""GPT-2: a decoder-only Transformer language model.

Attention Is All You Need (Vaswani et al., 2017) defines the block this file
stacks: scaled dot-product attention, several heads, a position-wise
feed-forward layer, residual connections, and a causal mask so position i
only reads positions 0..i.

Language Models are Few-Shot Learners (Brown et al., 2020, section 2.1)
uses the GPT-2 form of that block: LayerNorm on the sub-layer input,
depth-scaled residual initialization, and the usual left-to-right
language-modeling loss. GPT-3 then alternates dense and banded sparse
attention. This file is the dense causal model that description starts from.

Token ids are integers. GPT-2's byte-level BPE vocabulary has 50,257 tokens
and a 1,024-token context. GPT-3 Small is the same width and depth with a
2,048-token context (Brown et al., Table 2.1).

Shapes are batch-first: B batch, T time, C channels (n_embd).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Tanh approximation of GELU used by the GPT-2 code."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must divide evenly by n_head ({self.n_head})")

    @property
    def d_head(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def small(cls) -> GPT2Config:
        """12 layers, width 768, 12 heads, context 1024. 124,439,808 parameters.
        Radford et al. call this size 117M.
        """
        return cls()

    @classmethod
    def medium(cls) -> GPT2Config:
        return cls(n_layer=24, n_embd=1024, n_head=16)

    @classmethod
    def large(cls) -> GPT2Config:
        return cls(n_layer=36, n_embd=1280, n_head=20)

    @classmethod
    def xl(cls) -> GPT2Config:
        return cls(n_layer=48, n_embd=1600, n_head=25)

    @classmethod
    def gpt3_small(cls) -> GPT2Config:
        """Brown et al. Table 2.1: GPT-2 small with a 2048-token context (125,226,240 parameters)."""
        return cls(block_size=2048)

    @classmethod
    def tiny(cls) -> GPT2Config:
        """Small enough for a CPU smoke test."""
        return cls(vocab_size=128, block_size=64, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention.

    One head is Vaswani et al. equation 1:

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Heads are concatenated and mixed with an output projection. The lower
    triangle keeps token i from reading tokens after i (decoder mask,
    section 3.1). GPT-2 fuses the three input projections into ``c_attn``.
    """

    # register_buffer does not announce a type, so without this annotation
    # the checker treats the attribute as Tensor | Module.
    causal_mask: torch.Tensor

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # scores: (B, n_head, T, T). Scale keeps the dots from growing with d_k.
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        weights = self.attn_drop(torch.softmax(scores, dim=-1))
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward layer, inner width 4 * d_model.

    Vaswani et al. use ReLU here. GPT-2 uses GELU and keeps the 4x expansion.
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm residual block.

    GPT-2 (and GPT-3) normalizes the sub-layer input:

        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    The 2017 Transformer normalized after the residual add.
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """Autoregressive language model. Logits at position t predict the next token."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # The output projection is the token embedding matrix.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # Released GPT-2 code draws position embeddings with std 0.01.
        nn.init.normal_(self.wpe.weight, mean=0.0, std=0.01)
        # Residual projections are scaled by 1/sqrt(N), N = number of residual
        # layers. Each block adds two residual paths, so N = 2 * n_layer.
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        # Tied weights are one parameter. PyTorch yields that tensor once.
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx: (B, T) token ids. Position t reads idx[:, :t+1].
            targets: (B, T) ids aligned with ``idx``. The logit at t is trained
                to predict targets[:, t]. For next-token loss, pass the sequence
                shifted one step: ``model(idx[:, :-1], idx[:, 1:])``.
                Use -1 for positions that should be ignored.

        Returns:
            logits (B, T, vocab_size), and a scalar cross-entropy loss when
            ``targets`` is set.
        """
        _, T = idx.shape
        if T < 1 or T > self.config.block_size:
            raise ValueError(f"sequence length {T} is outside 1..{self.config.block_size}")

        positions = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Append ``max_new_tokens`` tokens, each conditioned on the prefix only."""
        training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                if temperature <= 0:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        k = min(top_k, logits.size(-1))
                        cutoff = torch.topk(logits, k).values[:, -1:]
                        logits = logits.masked_fill(logits < cutoff, float("-inf"))
                    next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
            return idx
        finally:
            self.train(training)


def _demo() -> None:
    torch.manual_seed(0)
    config = GPT2Config.tiny()
    model = GPT2(config)

    # A future token must not change logits at earlier positions.
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    logits, _ = model(idx)
    changed = idx.clone()
    changed[:, -1] = (changed[:, -1] + 1) % config.vocab_size
    logits_changed, _ = model(changed)
    if not torch.allclose(logits[:, :-1], logits_changed[:, :-1]):
        raise RuntimeError("causal mask failed: a later token changed an earlier logit")
    if torch.allclose(logits[:, -1], logits_changed[:, -1]):
        raise RuntimeError("the last position ignored its own token")

    # Next-token loss on one fixed batch. A wired-up model memorizes it.
    train_idx = torch.randint(0, config.vocab_size, (8, 32))
    x, y = train_idx[:, :-1], train_idx[:, 1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    first_loss = 0.0
    last_loss = 0.0
    for step in range(80):
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("expected a loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
        if step == 0:
            first_loss = last_loss
    if last_loss >= first_loss:
        raise RuntimeError(f"loss did not fall: {first_loss:.4f} -> {last_loss:.4f}")

    generated = model.generate(train_idx[:1, :4], max_new_tokens=8, temperature=0.8, top_k=10)
    print(f"tiny params: {model.num_parameters():,}")
    print(f"loss: {first_loss:.4f} -> {last_loss:.4f}")
    print(f"logits: {tuple(logits.shape)}  generated: {tuple(generated.shape)}")


if __name__ == "__main__":
    _demo()
