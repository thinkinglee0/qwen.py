import torch
from qwen.config import ModelConfig


class KVCache:
    data: list[tuple[torch.Tensor, torch.Tensor]]   # index: layer_index
    def __init__(self, n_layers, max_seqs, cache_len, n_kv_heads, head_dim, device, dtype):
        self.data = [
            (
                torch.zeros(max_seqs, cache_len, n_kv_heads, head_dim, device=device, dtype=dtype),
                torch.zeros(max_seqs, cache_len, n_kv_heads, head_dim, device=device, dtype=dtype),
            )
            for _ in range(n_layers)
        ]
        
def init_kv_cache2(config: ModelConfig, max_seqs: int, cache_len: int) -> KVCache:
    return KVCache(config.num_hidden_layers, max_seqs, cache_len,
                         config.num_key_value_heads, config.head_dim, config.device, config.dtype)

