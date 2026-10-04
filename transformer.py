"""Encoder-decoder Transformer from Attention Is All You Need.

Sequence layout throughout this file is (seq_len, batch, d_model).
Attention masks use (seq_q or 1, seq_k, batch or 1), where 0 or False
blocks that key.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_positional_encoding(d_model: int, max_len: int = 5000) -> torch.Tensor:
    """Fixed sinusoidal positions, Vaswani et al. section 3.5.

    PE[p, 2i] = sin(p / 10000^(2i / d_model))
    PE[p, 2i+1] = cos(p / 10000^(2i / d_model))

    Shape is (max_len, 1, d_model) so the encoding broadcasts over the batch.
    """
    encodings = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
    )
    encodings[:, 0::2] = torch.sin(position * div_term)
    odd_dims = encodings[:, 1::2].size(1)
    encodings[:, 1::2] = torch.cos(position * div_term[:odd_dims])
    return encodings.unsqueeze(1)


def subsequent_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Causal mask of shape (seq_len, seq_len, 1). True means the key may be read."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).unsqueeze(-1)


def clone_module_list(module: nn.Module, n: int) -> nn.ModuleList:
    """Independent copies so each layer has its own weights."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class FeedForward(nn.Module):
    """Position-wise network: max(0, x W_1 + b_1) W_2 + b_2."""

    def __init__(self, d_model: int, d_ff: int, dropout_prob: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.dropout(F.relu(self.layer1(x))))


class PrepareForMultiHeadAttention(nn.Module):
    """Project the last dimension and split it into heads."""

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_shape = x.shape[:-1]
        return self.linear(x).view(*head_shape, self.heads, self.d_k)


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model ({d_model}) must divide evenly by heads ({heads})")
        self.d_k = d_model // heads
        self.heads = heads

        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        # Softmax runs over the key sequence, which is dimension 1 of the scores.
        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn: torch.Tensor | None = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        # (seq_q, batch, heads, d_k), (seq_k, batch, heads, d_k) -> (seq_q, seq_k, batch, heads)
        return torch.einsum("ibhd,jbhd->ijbh", query, key)

    def prepare_mask(
        self, mask: torch.Tensor, query_shape: torch.Size, key_shape: torch.Size
    ) -> torch.Tensor:
        if mask.shape[0] not in (1, query_shape[0]):
            raise ValueError(f"mask query length {mask.shape[0]} does not match {query_shape[0]}")
        if mask.shape[1] != key_shape[0]:
            raise ValueError(f"mask key length {mask.shape[1]} does not match {key_shape[0]}")
        if mask.shape[2] not in (1, query_shape[1]):
            raise ValueError(f"mask batch {mask.shape[2]} does not match {query_shape[1]}")
        return mask.unsqueeze(-1)

    def forward(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len, batch_size, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query, key) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(self.softmax(scores))
        self.attn = attn.detach()

        # (seq_q, seq_k, batch, heads) x (seq_k, batch, heads, d_k) -> (seq_q, batch, heads, d_k)
        mixed = torch.einsum("ijbh,jbhd->ibhd", attn, value)
        return self.output(mixed.reshape(seq_len, batch_size, -1))


class EmbeddingsWithPositionalEncoding(nn.Module):
    # register_buffer does not announce a type. Without this, the checker
    # treats the attribute as Tensor | Module.
    positional_encodings: torch.Tensor

    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super().__init__()

        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.register_buffer("positional_encodings", get_positional_encoding(d_model, max_len))

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[: x.shape[0]].requires_grad_(False)
        return self.linear(x) * math.sqrt(self.d_model) + pe


class EmbeddingsWithLearnedPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[: x.shape[0]]
        return self.linear(x) * math.sqrt(self.d_model) + pe


class TransformerLayer(nn.Module):

    def __init__(
        self,
        *,
        d_model: int,
        self_attn: MultiHeadAttention,
        src_attn: MultiHeadAttention | None = None,
        feed_forward: FeedForward,
        dropout_prob: float,
    ):

        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_src_attn: nn.LayerNorm | None = None
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

        self.is_save_ff_input = False

    def forward(
        self,
        *,
        x: torch.Tensor,
        mask: torch.Tensor,
        src: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ):

        z = self.norm_self_attn(x)
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        x = x + self.dropout(self_attn)

        # Decoder layer: queries come from the target, keys and values from the encoder.
        if src is not None:
            if self.src_attn is None or self.norm_src_attn is None:
                raise RuntimeError("src was passed to a layer that has no source attention")
            z = self.norm_src_attn(x)
            attn_src = self.src_attn(query=z, key=src, value=src, mask=src_mask)
            x = x + self.dropout(attn_src)

        z = self.norm_ff(x)

        if self.is_save_ff_input:
            self.ff_input = z.clone()

        ff = self.feed_forward(z)
        x = x + self.dropout(ff)

        return x


class Encoder(nn.Module):

    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()

        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, mask: torch.Tensor):

        # Run through each transformer layer
        for layer in self.layers:
            x = layer(x=x, mask=mask)

        # Finally, normalize the vectors
        return self.norm(x)


# Transformer Decoder
class Decoder(nn.Module):

    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()

        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)

        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        # Run through each transformer layer
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)

        # Finally, normalize the vectors
        return self.norm(x)


# Generator
# This predicts the tokens and gives the lof softmax of those. You don't need this if you are using nn.CrossEntropyLoss .

class Generator(nn.Module):
    def __init__(self, n_vocab: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        return self.projection(x)


# Combined Encoder-Decoder
class EncoderDecoder(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module, generator: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

        # This was important from their code. Initialize parameters with Glorot / fan_avg.

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        # Run the source through encoder
        enc = self.encode(src, src_mask)
        # Run encodings and targets through decoder
        return self.decode(enc, src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def _demo() -> None:
    torch.manual_seed(0)
    d_model, heads, d_ff, n_layers = 32, 4, 64, 2
    n_vocab, src_len, tgt_len, batch = 50, 6, 5, 3

    def make_layer(with_source: bool) -> TransformerLayer:
        return TransformerLayer(
            d_model=d_model,
            self_attn=MultiHeadAttention(heads, d_model, dropout_prob=0.0),
            src_attn=MultiHeadAttention(heads, d_model, dropout_prob=0.0) if with_source else None,
            feed_forward=FeedForward(d_model, d_ff, dropout_prob=0.0),
            dropout_prob=0.0,
        )

    model = EncoderDecoder(
        encoder=Encoder(make_layer(False), n_layers),
        decoder=Decoder(make_layer(True), n_layers),
        src_embed=EmbeddingsWithPositionalEncoding(d_model, n_vocab),
        tgt_embed=EmbeddingsWithLearnedPositionalEncoding(d_model, n_vocab),
        generator=Generator(n_vocab, d_model),
    )
    model.eval()

    src = torch.randint(0, n_vocab, (src_len, batch))
    tgt = torch.randint(0, n_vocab, (tgt_len, batch))
    src_mask = torch.ones(1, src_len, 1)
    tgt_mask = subsequent_mask(tgt_len)

    hidden = model(src, tgt, src_mask, tgt_mask)
    logits = model.generator(hidden)
    changed = tgt.clone()
    changed[-1] = (changed[-1] + 1) % n_vocab
    hidden_changed = model(src, changed, src_mask, tgt_mask)
    if not torch.allclose(hidden[:-1], hidden_changed[:-1]):
        raise RuntimeError("decoder read a future target token")

    attn = model.decoder.layers[0].self_attn.attn
    if attn is None:
        raise RuntimeError("decoder self-attention was not stored")
    future = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool), diagonal=1)
    if torch.count_nonzero(attn[future]) != 0:
        raise RuntimeError("causal mask left weight on a future key")

    loss = F.cross_entropy(logits.reshape(-1, n_vocab), tgt.reshape(-1))
    loss.backward()
    print(f"hidden: {tuple(hidden.shape)} logits: {tuple(logits.shape)} loss: {loss.item():.4f}")


if __name__ == "__main__":
    _demo()
