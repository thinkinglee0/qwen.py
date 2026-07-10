import torch
from torch import nn

from qwen.cache import KVCache
from qwen.config import ModelConfig
from qwen.attention import Attention, AttentionMetadata
from qwen.mlp import MLP
from qwen.utils import RMSNorm


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_index: int):
        super().__init__()
        self.layer_index = layer_index

        self.self_attn = Attention(cfg, layer_index)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, hidden_states, cache: KVCache | None = None, causal_bias: torch.Tensor | None = None, past_len: int = 0):
        # pre-attention norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # self attention
        meta = AttentionMetadata(cache=cache, causal_bias=causal_bias, past_len=past_len)
        hidden_states = self.self_attn.forward(hidden_states, meta)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # fully connected, mlp
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, meta.cache
