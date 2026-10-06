import math 
from typing import Optional, List 
import torch 
from torch import nn 
from labml import tracker 


# https://nn.labml.ai/transformers/mha.html

class LayerHelper(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()

        # linear layer for linear transform 
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # number of heads
        self.heads = heads
        # number of dims in vectors in each head 
        self.d_k = d_k 

    def forward(self, x: torch.Tensor):
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)

        return x 



class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        self.d_k   = d_model//heads 
        self.heads = heads 

        self.query = LayerHelper(d_model, heads, self.d_k, bias=bias)
        self.key   = LayerHelper(d_model, heads, self.d_k, bias=bias)
        self.value = LayerHelper(d_model, heads, self.d_k, bias=True)

        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None 

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        return torch.einsum("ibh,jbh->ijbh", query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: torch.Size, key_shape: torch.Size):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[2]

        mask = mask.unsqueeze(-1)

        return mask

    def forward(self, *, 
                query: torch.Tensor,
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None ):
        
        seq_len, batch_size, _ = query.shape 

        if mask is not None: 
            mask = self.prepare_mask(mask, query.shape, key.shape)

        query = self.query(query)
        key   = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query, key)
        scores *= self.scale 

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.softmax(scores)

        attn = self.dropout(attn)

        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        self.attn = attn.detach()

        x = x.reshape(seq_len, batch_size, -1)

        return self.output(x)


