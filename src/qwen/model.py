import dataclasses
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging

from qwen.config import ModelConfig
from qwen.decode_layer import DecoderLayer
from qwen.attention import AttentionMetadata
from qwen.utils import RMSNorm
from qwen.sampling import apply_penalties2, sample2, SamplingMetadata

logger = logging.getLogger(__name__)


class QwenModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(self, input_ids: torch.Tensor, meta: AttentionMetadata) -> CausalLMOutputWithPast:
        # input_ids [T], hidden_states [T, hidden_size]
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, meta)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class QwenForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, max_seqs: int = 20, cache_len: int = 500):
        super().__init__()
        self.config = dataclasses.replace(cfg, weights=None, max_seqs=max_seqs, cache_len=cache_len)
        self.device = self.config.device     # resolved in config
        self.dtype = self.config.dtype

        self.model = QwenModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        missing, unexpected = self.load_state_dict(cfg.weights, strict=False)
        assert not unexpected, f"stale/renamed keys: {unexpected[:5]}"
        assert missing in ([], ["lm_head.weight"]), f"missing: {missing}"

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, meta: AttentionMetadata) -> CausalLMOutputWithPast:
        # input_ids: packed varlen, shape [T]
        hidden_states = self.model.forward(input_ids, meta)

        return hidden_states    # shape [T, hidden_size]
    
    def compute_logits(self, hidden_states: torch.Tensor):
        return self.lm_head(hidden_states)  # shape [T, vocab_size]

    def sampler(self, logits, prompt_tokens, output_tokens, sampling_meta: SamplingMetadata | None = None):
        if sampling_meta is None:
            B, _ = logits.size()
            sampling_meta = SamplingMetadata(config=self.config, bsz=B)
        
        logits = apply_penalties2(logits, prompt_tokens, output_tokens, sampling_meta, self.config.vocab_size)
        if self.config.do_sample:
            next_tokens = sample2(logits, sampling_meta)
        else:
            next_tokens = logits.argmax(dim=-1)

        return next_tokens

