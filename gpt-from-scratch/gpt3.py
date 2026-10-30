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

Shapes are batch-first: B batch, T time, C channels (n_embd)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPT3Config:
    """GPT-2 small dimensions and regularization settings."""

    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


class CasualSelfAttention(nn.Module):
    """Let each token attend to itself and earlier tokens, but not future tokens."""

    def __init__(self, config):
        super().__init__()  # Initialize PyTorch's base module machinery.
        self.n_heads = config.n_heads  # Store the number of parallel attention heads.
        self.head_dim = config.d_model // config.n_heads  # Compute the feature width of each head.
        self.dropout = config.dropout  # Store the attention dropout probability.

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, past_key_value=None, use_cache=False):
        """Compute causal attention for an input shaped (batch, sequence, features)."""
        B, T, C = x.shape  # Read batch size, sequence length, and model width.

        q, k, v = self.qkv(x).chunk(3, dim=-1)  # Split the combined projection into query, key, and value tensors.

        def split_heads(t):
            """Reshape a projected tensor into (batch, heads, sequence, head features)."""
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # Put heads before sequence for attention.

        q, k, v = map(split_heads, (q, k, v))  # Format all three tensors for multi-head attention.

        #cache layout: (B, H, seq_len, head_dim) 
        past_len = 0

        if past_key_value is not None:
            past_k, past_v = past_key_value 
            past_len = past_k.size(-2)

            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        total_len = k.size(-2)

        if past_len == 0:
            # Normal causal attention during training or prefill.
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # Query positions are past_len ... past_len + T - 1.
            # Key positions are 0 ... total_len - 1.
            query_positions = torch.arange(past_len, total_len, device=x.device)[:, None]

            key_positions = torch.arange(total_len, device=x.device)[None, :]

            # True means this key position may be attended to.
            causal_mask = key_positions <= query_positions

            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=causal_mask[None, None, :, :],
                is_causal=False,
                dropout_p=self.dropout if self.training else 0.0,
            )

        y = (y.transpose(1, 2).contiguous().view(B, T, C))

        present = (k, v) if use_cache else None

        return self.resid_dropout(self.out_proj(y)), present


class MLP(nn.Module):
    """Apply a nonlinear, two-layer transformation independently to each token."""

    def __init__(self, config):  # Initialize layers using the model configuration.
        super().__init__()  # Initialize PyTorch's base module machinery.
        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)  # Expand token features to four times the model width.
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)  # Project the expanded features back to model width.
        self.dropout = nn.Dropout(config.dropout)  # Configure dropout for regularization.

    def forward(self, x):  # Compute the MLP output for the input token representations.
        x = self.fc1(x)  # Map features from model width to the wider hidden dimension.
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)  # Map hidden features back to model width.
        return self.dropout(x)  # Apply dropout and return the transformed features.


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CasualSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x, past_key_value=None, use_cache=None):
        attn_out, present = self.attn(
            self.ln1(x),
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))

        return x, present


class GPT3(nn.Module):
    """GPT-2-compatible decoder, retaining the original project class name."""

    def __init__(self, config):
        super().__init__()
        self.config = config 

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight 

        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / (2 * config.n_layers) ** 0.5,
                )

    @classmethod
    def from_pretrained(cls, model_name="openai-community/gpt2", **model_kwargs):
        """Load GPT-2 weights into this implementation."""
        from transformers import GPT2LMHeadModel

        pretrained = GPT2LMHeadModel.from_pretrained(model_name, **model_kwargs)
        hf_config = pretrained.config
        if hf_config.model_type != "gpt2":
            raise ValueError(
                f"Expected a GPT-2 checkpoint, got {hf_config.model_type!r}"
            )

        config = GPT3Config(
            vocab_size=hf_config.vocab_size,
            max_seq_len=hf_config.n_positions,
            d_model=hf_config.n_embd,
            n_heads=hf_config.n_head,
            n_layers=hf_config.n_layer,
            dropout=hf_config.resid_pdrop,
            layer_norm_eps=hf_config.layer_norm_epsilon,
        )
        model = cls(config)
        parameter_mapping = {
            "token_embedding.weight": "transformer.wte.weight",
            "position_embedding.weight": "transformer.wpe.weight",
            "ln_f.weight": "transformer.ln_f.weight",
            "ln_f.bias": "transformer.ln_f.bias",
            "lm_head.weight": "lm_head.weight",
        }
        transposed_parameters = set()
        for index in range(config.n_layers):
            source_prefix = f"transformer.h.{index}."
            target_prefix = f"blocks.{index}."
            layer_mapping = {
                "ln1.weight": "ln_1.weight",
                "ln1.bias": "ln_1.bias",
                "attn.qkv.weight": "attn.c_attn.weight",
                "attn.qkv.bias": "attn.c_attn.bias",
                "attn.out_proj.weight": "attn.c_proj.weight",
                "attn.out_proj.bias": "attn.c_proj.bias",
                "ln2.weight": "ln_2.weight",
                "ln2.bias": "ln_2.bias",
                "mlp.fc1.weight": "mlp.c_fc.weight",
                "mlp.fc1.bias": "mlp.c_fc.bias",
                "mlp.fc2.weight": "mlp.c_proj.weight",
                "mlp.fc2.bias": "mlp.c_proj.bias",
            }
            for target_name, source_name in layer_mapping.items():
                target_key = target_prefix + target_name
                parameter_mapping[target_key] = source_prefix + source_name
                if target_name.endswith(".weight") and target_name not in {
                    "ln1.weight",
                    "ln2.weight",
                }:
                    transposed_parameters.add(target_key)

        checkpoint_state = pretrained.state_dict()
        model_state = {}
        for target_name, source_name in parameter_mapping.items():
            weight = checkpoint_state[source_name]
            if target_name in transposed_parameters:
                weight = weight.T
            model_state[target_name] = weight
        model.load_state_dict(model_state, strict=True)

        model.eval()
        return model

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, past_key_values=None, use_cache=False):
        B, T = input_ids.shape

        if past_key_values is None:
            past_len = 0
        else:
            past_len = past_key_values[0][0].size(-2)

        if past_len + T > self.config.max_seq_len:
            raise ValueError("Context exceeds max_seq_len")

        # Positions must continue from the cached sequence length.
        positions = torch.arange(
            past_len,
            past_len + T,
            device=input_ids.device,
        )

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)[None, :, :]
        )

        presents = []

        for i, block in enumerate(self.blocks):
            layer_past = (
                None if past_key_values is None
                else past_key_values[i]
            )

            x, present = block(
                x,
                past_key_value=layer_past,
                use_cache=use_cache,
            )

            if use_cache:
                presents.append(present)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(
                    -1, self.config.vocab_size
                ),
                labels[:, 1:].contiguous().view(-1),
            )

        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": tuple(presents) if use_cache else None,
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens):
        self.eval()

        if input_ids.size(1) + max_new_tokens > self.config.max_seq_len:
            raise ValueError(
                "Prompt plus generated tokens exceeds max_seq_len"
            )

        # Prefill: process the complete prompt once.
        output = self(input_ids, use_cache=True)
        cache = output["past_key_values"]
        logits = output["logits"][:, -1, :]

        for step in range(max_new_tokens):
            # Greedy decoding.
            next_token = logits.argmax(
                dim=-1, keepdim=True
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

            if step == max_new_tokens - 1:
                break

            # Process only the new token using the existing cache.
            output = self(
                next_token,
                past_key_values=cache,
                use_cache=True,
            )

            cache = output["past_key_values"]
            logits = output["logits"][:, -1, :]

        return input_ids