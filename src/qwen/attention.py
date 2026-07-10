import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple, Union
import math

from qwen.config import ModelConfig
from qwen.rope import BaseRoPE
from qwen.cache import KVCache
from qwen.rope import init_rope

logger = logging.getLogger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    
    bsz, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, slen, head_dim).contiguous().view(bsz, num_key_value_heads * n_rep, slen, head_dim)
    return hidden_states

@dataclass
class AttentionMetadata:
    cache: KVCache | None = None
    causal_bias: torch.Tensor | None = None
    past_len: int = 0


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.rope = init_rope(cfg)

        self.num_query_heads = cfg.num_attention_heads
        self.num_key_value_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.attn_dim = cfg.hidden_size

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor, meta: AttentionMetadata) -> Tensor:
        bsz, q_len, _ = hidden_states.size()

        # projection
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # rope
        query_states, key_states = self.rope(query_states, key_states, meta.past_len)

        #kv cache
        if meta.cache is not None:
            keys, vals = meta.cache[self.layer_index]
            key_states = torch.cat([keys, key_states], 2)
            value_states = torch.cat([vals, value_states], 2)
            meta.cache[self.layer_index] = (key_states, value_states)  # upate cache

        # repeat k,v for grouped-query attention (GQA)
        if self.num_query_heads % self.num_key_value_heads != 0:
            raise ValueError(f"num_query_heads ({self.num_query_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})")
        n_rep = self.num_query_heads // self.num_key_value_heads
    
        key_states = repeat_kv(key_states, n_rep)
        value_states = repeat_kv(value_states, n_rep)

        # attention
        attn_scores = query_states @ key_states.transpose(-2, -1) /  math.sqrt(self.head_dim)
        attn_scores = attn_scores + meta.causal_bias

        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        hidden_states = attn_weights @ value_states

        # output projection
        hidden_states = hidden_states.transpose(1, 2).contiguous().reshape(bsz, q_len, self.attn_dim)
        hidden_states = self.o_proj(hidden_states)

        return hidden_states

