import dataclasses
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging

from qwen.config import ModelConfig
from qwen.decode_layer import DecoderLayer
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

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(self, input_ids: torch.Tensor, cache: KVCache | None = None) -> CausalLMOutputWithPast:
        hidden_states = self.embed_tokens(input_ids)    # [bsz, seq_len, hidden_size]
        bsz, q_len, _ = hidden_states.size()
        past_len = 0 if cache is None else cache[0][0].shape[2]

        causal_bias = bottom_right_causal_bias(q_len, q_len+past_len, hidden_states.device, hidden_states.dtype)

        for layer in self.layers:
            hidden_states, cache = layer.forward(hidden_states, cache, causal_bias, past_len)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class QwenForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = dataclasses.replace(cfg, weights=None)
        self.device = self.config.device     # resolved in config
        self.dtype = self.config.dtype

        self.model = QwenModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        missing, unexpected = self.load_state_dict(cfg.weights, strict=False)
        assert not unexpected, f"stale/renamed keys: {unexpected[:5]}"
        assert missing in ([], ["lm_head.weight"]), f"missing: {missing}"

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, cache: KVCache | None = None) -> CausalLMOutputWithPast:
        hidden_states = self.model.forward(input_ids, cache)

        logits = self.lm_head(hidden_states[:, -1, :])

        # output
        output = CausalLMOutputWithPast()
        if cache is not None:
            output.past_key_values = tuple(cache)
        output.logits = logits
        return output
