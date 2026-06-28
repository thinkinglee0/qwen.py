import torch

from config import ModelConfig

def apply_rotary(x, cos, sin):
    x1 = x[..., :x.shape[-1] // 2]      # shape [1, 1, seq_len, d/2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)    # shape [1, 1, seq_len, d]

class BaseRoPE():
    def __init__(self, dim: int, max_seq_len: int, base: float = 1_000_000.0, fixed=True):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.fixed_max_seq_len = fixed
        self.base = base
        self._build_cache(max_seq_len)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        return 1.0 / base ** (torch.arange(0, self.dim, 2).float() / self.dim)
    
    def _build_cache(self, seq_len: int, base: float | None = None):
        inv_freq = self._compute_inv_freq(base or self.base)
        t = torch.arange(seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.cos_cached = freqs.cos()[None, None]   # shape [1, 1, seq_len, d/2]
        self.sin_cached = freqs.sin()[None, None]

    def __call__(self, q, k, offset: int = 0):
        T = q.shape[2]
        if self.fixed_max_seq_len:
            assert offset + T <= self.cos_cached.shape[2], \
                f"seq overflow: offset={offset}, T={T}, max={self.max_seq_len}"
        cos = self.cos_cached[..., offset:offset+T, :]
        sin = self.sin_cached[..., offset:offset+T, :]

        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

class DefaultRoPE(BaseRoPE):
    pass
    
class LinearRoPE(BaseRoPE):
    def __init__(self, dim: int, max_seq_len: int, base: float = 1_000_000.0, scale: float = 1):
        self.scale = scale
        super().__init__(dim, max_seq_len, base)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        return super()._compute_inv_freq(base) / self.scale

class DynamicNTKRoPE(BaseRoPE):
    def __init__(self, dim: int, max_seq_len: int, base: float = 1_000_000.0, scale: float = 1):
        self.scale = scale
        self._cached_seq_len = 0
        super().__init__(dim, max_seq_len, base, fixed=False)
        self._cached_seq_len = max_seq_len  # update _cached_seq_len after building cos/sin cache with length of max_seq_len

    def __call__(self, q, k, offset: int = 0):
        current_seq_len = q.shape[2] + offset
        
        if current_seq_len > self._cached_seq_len:
            # base' = base * (scale * L / max - (scale - 1))^(d/(d-2))
            ratio = current_seq_len * self.scale / self.max_seq_len - (self.scale-1)    # better to use current_seq_len instead of new_cache_len, latter will lower the precision
            base_new =  self.base * (ratio ** (self.dim / (self.dim-2)))

            new_cache_len = max(self._cached_seq_len * 2, current_seq_len)
            self._build_cache(new_cache_len, base=base_new)
            self._cached_seq_len = new_cache_len

        return super().__call__(q, k, offset)
    
def init_rope(config: ModelConfig) -> BaseRoPE:
    match config.rope_scaling.get("rope_type", "default"):
        case "default":
            return DefaultRoPE(config.head_dim, config.max_position_embeddings, config.rope_theta)
        case "linear":
            return LinearRoPE(config.head_dim, config.max_position_embeddings, config.rope_theta, scale=config.rope_scaling["factor"])
        case "dynamic":
            return DynamicNTKRoPE(config.head_dim, config.max_position_embeddings, config.rope_theta, scale=config.rope_scaling["factor"])
        case _:
            return None