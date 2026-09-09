import torch
import logging
from torch import nn

from qwen.config import ModelConfig

logger = logging.getLogger(__name__)

def apply_rotary(x, cos, sin):
    x1 = x[..., :x.shape[-1] // 2]      # [T, H, D/2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)    # [T, H, D]

_compiled_apply_rotary = None   # only compiled once

def get_apply_rotary(compile: bool):
    # torch.compile fuses the whole bf16 pointwise chain, keeps the intermediates in fp32 and
    # rounds only once at the end; HF's eager path rounds after every op. The two drift by
    # ~1 bf16 ulp per layer, which compounds over 24 layers into a ~0.28 logits gap.
    # So HF-parity tests must run eager; serving can take the compiled path's speed.
    global _compiled_apply_rotary
    if not compile:
        logger.info("Using eager path for apply_rotary")
        return apply_rotary
    if _compiled_apply_rotary is None:
        _compiled_apply_rotary = torch.compile(apply_rotary, dynamic=True, fullgraph=True)

    logger.info("Using compiled path for apply_rotary")
    return _compiled_apply_rotary

class BaseRoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, compile_rope: bool, base: float = 1_000_000.0, fixed=True):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.fixed_max_seq_len = fixed
        self.base = base
        self._build_cache(max_seq_len)
        self._apply_rotary = get_apply_rotary(compile_rope)


    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        return 1.0 / base ** (torch.arange(0, self.dim, 2).float() / self.dim)
    
    def _build_cache(self, seq_len: int, base: float | None = None):
        inv_freq = self._compute_inv_freq(base or self.base)
        t = torch.arange(seq_len).float()
        freqs = torch.outer(t, inv_freq)
        # self.cos_cached = freqs.cos()[:, None, :]   # shape [seq_len, H=1, d/2]
        # self.sin_cached = freqs.sin()[:, None, :]
        self.register_buffer("cos_cached", freqs.cos()[:, None, :], persistent=False)   # shape [seq_len, H=1, d/2]
        self.register_buffer("sin_cached", freqs.sin()[:, None, :], persistent=False)

    def forward(self, q, k, position_ids):
        # q,k [T, H, D]
        # position_ids [T]
        # if self.fixed_max_seq_len:
        #     assert position_ids.max() < self.cos_cached.shape[0], \
        #         f"seq overflow: position_ids={position_ids.max()}, max={self.max_seq_len}"
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]

        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

class DefaultRoPE(BaseRoPE):
    pass
    
class LinearRoPE(BaseRoPE):
    def __init__(self, dim: int, max_seq_len: int, compile_rope: bool, base: float = 1_000_000.0, scale: float = 1):
        self.scale = scale
        super().__init__(dim, max_seq_len, compile_rope=compile_rope, base=base)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        return super()._compute_inv_freq(base) / self.scale

# class DynamicNTKRoPE(BaseRoPE):
#     def __init__(self, dim: int, max_seq_len: int, base: float = 1_000_000.0, scale: float = 1):
#         self.scale = scale
#         self._cached_seq_len = 0
#         super().__init__(dim, max_seq_len, base, fixed=False)
#         self._cached_seq_len = max_seq_len  # update _cached_seq_len after building cos/sin cache with length of max_seq_len

#     def __call__(self, q, k, position_ids):
#         current_seq_len = q.shape[2] + offset
        
#         if current_seq_len > self._cached_seq_len:
#             # base' = base * (scale * L / max - (scale - 1))^(d/(d-2))
#             ratio = current_seq_len * self.scale / self.max_seq_len - (self.scale-1)    # better to use current_seq_len instead of new_cache_len, latter will lower the precision
#             base_new =  self.base * (ratio ** (self.dim / (self.dim-2)))

#             new_cache_len = max(self._cached_seq_len * 2, current_seq_len)
#             self._build_cache(new_cache_len, base=base_new)
#             self._cached_seq_len = new_cache_len

#         return super().__call__(q, k, offset)
    
def init_rope(config: ModelConfig) -> BaseRoPE:
    assert config.rope_scaling, "rope_scaling must be provided in config"
    assert config.compile_rope is not None, "compile_rope must be provided in config"

    match config.rope_scaling.get("rope_type", "default"):
        case "default":
            return DefaultRoPE(config.head_dim, config.max_position_embeddings, compile_rope=config.compile_rope, base=config.rope_theta)
        case "linear":
            return LinearRoPE(config.head_dim, config.max_position_embeddings, compile_rope=config.compile_rope, base=config.rope_theta, scale=config.rope_scaling["factor"])
        # case "dynamic":
        #     return DynamicNTKRoPE(config.head_dim, config.max_position_embeddings, config.rope_theta, scale=config.rope_scaling["factor"])
        case _:
            raise ValueError(f"unsupported rope_type: {config.rope_scaling.get('rope_type')}")