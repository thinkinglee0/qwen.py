import torch
from qwen.config import ModelConfig

# index: cache[layer_idx] = (k, v)
# k, v shape: (B, n_kv_heads, seq_len_so_far, head_dim)
KVCache = list[tuple[torch.Tensor, torch.Tensor]]

def init_kv_cache2(config: ModelConfig,
                   bsz: int,
                   device: torch.device,
                   dtype: torch.dtype):
    return init_kv_cache(config.num_hidden_layers, bsz, config.num_key_value_heads, config.head_dim, device, dtype)

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

