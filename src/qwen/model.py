import dataclasses
import math
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging

from qwen.config import ModelConfig
from qwen.decode_layer import DecoderLayer
from qwen.rope import init_rope
from qwen.cache import KVCache
from qwen.utils import RMSNorm

logger = logging.getLogger(__name__)

def bottom_right_causal_bias(q_len: int, k_len: int, device: torch.device, dtype: torch.dtype,) -> torch.Tensor:
    # position_ids: the i-th query token maps to global position (k_len - q_len + i)
    # allow attending to j <= k_len - q_len + i, i.e. mask out j > k_len - q_len + i  <=>  j - i >= k_len - q_len + 1
    mask = torch.full((q_len, k_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=k_len - q_len + 1)
    return mask[None, None]  # (1, 1, q_len, k_len)

class QwenModel(nn.Module):
    def __init__(self, config: ModelConfig, max_bsz: int = 1):
        super().__init__()

        # isolate from the session-scoped pytest fixture `target_config` to avoid cross-test mutation
        self.config = dataclasses.replace(config, weights=None)
        self.config.weights = config.weights
        self.weights = config.weights

        self.device = self.config.device     # resolved in config
        self.dtype = self.config.dtype

        self.rope = init_rope(self.config)

        self.layers = nn.ModuleList(
            [DecoderLayer(self.config, layer_idx, self.rope) for layer_idx in range(self.config.num_hidden_layers)]
        )

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weights['model.embed_tokens.weight'][input_ids]

    def unembed(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # logits
        return hidden_states @ self.weights['lm_head.weight'].transpose(-2, -1)

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # lm head, hidden_states shape [bsz, seq_len, hidden_size]
        # last_hidden_states shape [bsz, hidden_size]
        last_hidden_states = RMSNorm(hidden_states[:, -1, :], self.weights['model.norm.weight'], eps=self.config.rms_norm_eps)
        logits = self.unembed(last_hidden_states)
        return logits

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, cache: KVCache | None = None) -> CausalLMOutputWithPast:
        hidden_states = self.embed_tokens(input_ids)    # [bsz, seq_len, hidden_size]
        bsz, q_len, _ = hidden_states.size()
        past_len = 0 if cache is None else cache[0][0].shape[2]

        causal_bias = bottom_right_causal_bias(q_len, q_len+past_len, hidden_states.device, hidden_states.dtype)

        for layer in self.layers:
            hidden_states, cache = layer.forward(hidden_states, cache, causal_bias, past_len)

        logits = self.lm_head(hidden_states)

        # output
        output = CausalLMOutputWithPast()
        if cache is not None:
            output.past_key_values = tuple(cache)
        output.logits = logits
        return output


