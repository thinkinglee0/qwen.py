import math
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from pathlib import Path
import logging

from config import load_qwen_weights, ModelConfig
from rope import init_rope
from utils import resolve_device, default_dtype

# index: cache[layer_idx] = (k, v)
# k, v shape: (B, n_kv_heads, seq_len_so_far, head_dim)
KVCache = list[tuple[torch.Tensor, torch.Tensor]]

logger = logging.getLogger(__name__)

def init_kv_cache(n_layers: int,
                  bsz: int,
                  n_kv_heads: int,
                  head_dim: int,
                  device: torch.device,
                  dtype: torch.dtype,
                  ):
    return [
        (
            # seq_len = 0
            torch.zeros(bsz, n_kv_heads, 0, head_dim, device=device, dtype=dtype),
            torch.zeros(bsz, n_kv_heads, 0, head_dim, device=device, dtype=dtype),
        )
        for _ in range(n_layers)
    ]

def RMSNorm(hidden_states : torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    in_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)     # upcast to float32
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(in_dtype)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    
    bsz, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, slen, head_dim).contiguous().view(bsz, num_key_value_heads * n_rep, slen, head_dim)
    return hidden_states

def make_causal_mask(q_len: int, k_len: int, device: torch.device, dtype: torch.dtype,) -> torch.Tensor:
    # position_ids: the i-th query token maps to global position (k_len - q_len + i)
    # allow attending to j <= k_len - q_len + i, i.e. mask out j > k_len - q_len + i  <=>  j - i >= k_len - q_len + 1
    mask = torch.full((q_len, k_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=k_len - q_len + 1)
    return mask[None, None]  # (1, 1, q_len, k_len)

class QwenModel(nn.Module):
    def __init__(self, model_dir: str | Path, max_bsz: int = 1):
        super().__init__()
        self.config = ModelConfig.from_pretrained(model_dir)
        self.weights = load_qwen_weights(self.config, model_dir)

        self.rope = init_rope(self.config)

        # Initialize model parameters based on config
        self.attn_dim = self.config.hidden_size
        self.num_query_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.head_dim = self.config.head_dim

        self.max_bsz = max_bsz
        self.device = resolve_device()
        self.dtype = default_dtype(self.device)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weights['model.embed_tokens.weight'][input_ids]

    def unembed(self, hidden_states: torch.torch) -> torch.Tensor:
        # logits
        return hidden_states @ self.weights['lm_head.weight'].transpose(-2, -1)
    
    def attention(self, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        attn_scores = query_states @ key_states.transpose(-2, -1) /  math.sqrt(self.head_dim)
        attn_scores = attn_scores + causal_mask

        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = attn_weights @ value_states
        return attn_output

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, cache: KVCache | None = None) -> CausalLMOutputWithPast:
        hidden_states = self.embed_tokens(input_ids)    # [bsz, seq_len, hidden_size]
        bsz, q_len, _ = hidden_states.size()

        past_len = 0 if cache is None else cache[0][0].shape[2]
        if cache is None:
            past_len = 0
            cache = init_kv_cache(self.config.num_hidden_layers, bsz, self.config.num_key_value_heads, self.config.head_dim,
                              hidden_states.device, self.dtype)
        else:
            past_len = cache[0][0].shape[2]
        causal_mask = make_causal_mask(q_len, q_len+past_len, hidden_states.device, hidden_states.dtype)

        for layer in range(self.config.num_hidden_layers):
            # pre-attention norm
            residual = hidden_states
            hidden_states = RMSNorm(hidden_states, self.weights[f'model.layers.{layer}.input_layernorm.weight'], eps=self.config.rms_norm_eps)

            # projection
            query_states = hidden_states @ self.weights[f'model.layers.{layer}.self_attn.q_proj.weight'].transpose(-2, -1) + self.weights[f'model.layers.{layer}.self_attn.q_proj.bias']
            key_states = hidden_states @ self.weights[f'model.layers.{layer}.self_attn.k_proj.weight'].transpose(-2, -1) + self.weights[f'model.layers.{layer}.self_attn.k_proj.bias']
            value_states = hidden_states @ self.weights[f'model.layers.{layer}.self_attn.v_proj.weight'].transpose(-2, -1) + self.weights[f'model.layers.{layer}.self_attn.v_proj.bias']

            # reshape for multi-head attention
            query_states = query_states.view(bsz, q_len, self.num_query_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # rope
            query_states, key_states = self.rope(query_states, key_states, past_len)

            #kv cache
            keys, vals = cache[layer]
            key_states = torch.cat([keys, key_states], 2)
            value_states = torch.cat([vals, value_states], 2)
            cache[layer] = (key_states, value_states)  # upate cache

            # repeat k,v for grouped-query attention (GQA)
            if self.num_query_heads % self.config.num_key_value_heads != 0:
                raise ValueError(f"num_query_heads ({self.num_query_heads}) must be divisible by num_key_value_heads ({self.config.num_key_value_heads})")
            n_rep = self.num_query_heads // self.config.num_key_value_heads
        
            key_states = repeat_kv(key_states, n_rep)
            value_states = repeat_kv(value_states, n_rep)

            # attention
            hidden_states = self.attention(query_states, key_states, value_states, causal_mask)
            # return key_states, value_states

            # output projection
            hidden_states = hidden_states.transpose(1, 2).contiguous().reshape(bsz, q_len, self.attn_dim)
            hidden_states = hidden_states @ self.weights[f'model.layers.{layer}.self_attn.o_proj.weight'].transpose(-2, -1)

            #residual connection
            hidden_states = residual + hidden_states

            # pre-mlp norm
            residual = hidden_states
            hidden_states = RMSNorm(hidden_states, self.weights[f'model.layers.{layer}.post_attention_layernorm.weight'], eps=self.config.rms_norm_eps)

            # mlp
            gate = hidden_states @ self.weights[f'model.layers.{layer}.mlp.gate_proj.weight'].transpose(-2, -1)
            up = hidden_states @ self.weights[f'model.layers.{layer}.mlp.up_proj.weight'].transpose(-2, -1)
            mlp_act = torch.nn.functional.silu(gate) * up
            hidden_states = mlp_act @ self.weights[f'model.layers.{layer}.mlp.down_proj.weight'].transpose(-2, -1)

            # residual connection
            hidden_states = residual + hidden_states

        hidden_states = RMSNorm(hidden_states, self.weights['model.norm.weight'], eps=self.config.rms_norm_eps)
        logits = self.unembed(hidden_states)

        # output
        output = CausalLMOutputWithPast()
        output.past_key_values = tuple(cache)
        output.logits = logits
        return output

    def sample(self, logits: torch.Tensor):
        last_logit = logits[:, -1, :]
        latest_tokens = last_logit.argmax(1, keepdim=True)
        return latest_tokens

    def generate(self, input_ids: torch.Tensor, max_len: int = 300) -> torch.Tensor:
        bsz, seq_len = input_ids.size()
        if bsz > 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        
        cache = init_kv_cache(self.config.num_hidden_layers, bsz, self.config.num_key_value_heads, self.config.head_dim,
                              self.device, self.dtype)
        latest_token_id = input_ids[0, -1]  # todo: only for bsz=1
        output_tokens = input_ids.clone()
        latest_tokens = input_ids
        while latest_token_id != self.config.eos_token_id and output_tokens.size(1) < min(self.config.max_position_embeddings, seq_len+max_len):
            output = self.forward(latest_tokens, list(cache))
            cache = output.past_key_values
            latest_tokens = self.sample(output.logits)
            output_tokens = torch.cat([output_tokens, latest_tokens], -1)
            latest_token_id = latest_tokens[0, 0]

        return output_tokens

