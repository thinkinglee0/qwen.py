import torch
from torch import nn

from qwen.cache import KVCache
from qwen.config import ModelConfig
from qwen.attention import Attention, AttentionMetadata
from qwen.mlp import MLP
from qwen.rope import BaseRoPE
from qwen.utils import RMSNorm


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_index: int, rope: BaseRoPE):
        super().__init__()
        self.config = cfg
        self.weights = cfg.weights
        self.layer_index = layer_index

        self.self_attn = Attention(cfg, layer_index, rope)
        self.mlp = MLP(cfg, layer_index)

    def forward(self, hidden_states, cache: KVCache | None = None, causal_bias: torch.Tensor | None = None, past_len: int = 0):
        # pre-attention norm
        residual = hidden_states
        hidden_states = RMSNorm(hidden_states, self.weights[f'model.layers.{self.layer_index}.input_layernorm.weight'], eps=self.config.rms_norm_eps)

        # self attention
        meta = AttentionMetadata(cache=cache, causal_bias=causal_bias, past_len=past_len)
        hidden_states = self.self_attn.forward(hidden_states, meta)
        hidden_states = residual + hidden_states

        # pre-mlp/post-attention norm
        residual = hidden_states
        hidden_states = RMSNorm(hidden_states, self.weights[f'model.layers.{self.layer_index}.post_attention_layernorm.weight'], eps=self.config.rms_norm_eps)

        # fully connected, mlp
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, meta.cache
