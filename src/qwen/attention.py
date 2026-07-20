import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import logging
import itertools

from qwen.config import ModelConfig
from qwen.rope import BaseRoPE
from qwen.cache import KVCache
from qwen.rope import init_rope

# attention backend selection — resolved once at import
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache  # for A10
    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    HAS_FLASH_ATTN = False

logger = logging.getLogger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    
    # q, k, v: (B, H, S, D)
    bsz, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, slen, head_dim).contiguous().view(bsz, num_key_value_heads * n_rep, slen, head_dim)
    return hidden_states

def _bottom_right_causal_bias(q_len: int, k_len: int, device: torch.device, dtype: torch.dtype,) -> torch.Tensor:
    # position_ids: the i-th query token maps to global position (k_len - q_len + i)
    # allow attending to j <= k_len - q_len + i, i.e. mask out j > k_len - q_len + i  <=>  j - i >= k_len - q_len + 1
    mask = torch.full((q_len, k_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=k_len - q_len + 1)
    return mask[None, None]  # (1, 1, q_len, k_len)


@dataclass
class AttentionMetadata:
    is_prefill: bool
    cache: KVCache

    # query side
    cu_seqlens_q: Tensor | None = None   # (num_seqs + 1,) prefix-sum of query lengths
    max_seqlen_q: int | None = None

    # key side
    cu_seqlens_k: Tensor | None = None   # (num_seqs,) per-request KV length
    max_seqlen_k: int | None = None

    # same info as cu_seqlens_k, plain (non-cumulative) form for flash_attn_with_kvcache
    cache_seqlens: Tensor | None = None

    position_ids: Tensor | None = None
    slot_mapping: Tensor | None = None

    # debug cache issue
    debug_k_list: list[Tensor] | None = None
    debug_v_list: list[Tensor] | None = None

def pack_sequences(seqs, device, cache_len):
    # seqs: list of variable-length id sequences. Accepts list[int] (raw tokenizer
    # output) or 1D LongTensor; normalize once at the boundary.
    ids_per_seq = [s.tolist() if torch.is_tensor(s) else list(s) for s in seqs]
    lens = [len(s) for s in ids_per_seq]

    # Contiguous cache: each seq owns a fixed [cache_len] slab. Overflow would
    # silently corrupt the neighbour's KV, so gate it here.
    if max(lens) > cache_len:
        raise ValueError(f"seq len {max(lens)} exceeds cache_len {cache_len}")

    max_seqlen = max(lens)                    # plain int, no .max() -> on CPU, no D2H sync

    packed_ids = torch.tensor(
        list(itertools.chain.from_iterable(ids_per_seq)), device=device, dtype=torch.long
    )                                                                    # [T]

    # cu_seqlens MUST be int32 for flash_attn_varlen_func. Build on CPU to dodge
    # cumsum's int32 -> int64 integer_upcast entirely.
    cu_seqlens = torch.tensor(
        list(itertools.accumulate(lens, initial=0)), device=device, dtype=torch.int32
    )                                                                    # [B+1]

    lengths = torch.tensor(lens, device=device, dtype=torch.int32)       # [B]

    # Position ids RESTART at 0 per segment — RoPE-critical under packed layout.
    position_ids = torch.tensor(
        [p for l in lens for p in range(l)], device=device, dtype=torch.long
    )                                                                    # [T]

    doc_id = torch.tensor(
        [i for i, l in enumerate(lens) for _ in range(l)], device=device, dtype=torch.long
    )                                                                    # [T]

    slot_mapping = doc_id * cache_len + position_ids                     # [T], on GPU

    return packed_ids, cu_seqlens, position_ids, slot_mapping, max_seqlen, lengths

def build_prefill_metadata(seqs, cache: KVCache, device, cache_len):
    packed_ids, cu_seqlens, position_ids, slot_mapping, max_seqlen, lengths = \
        pack_sequences(seqs, device, cache_len)
    return packed_ids, AttentionMetadata(
        is_prefill=True, cache=cache,
        position_ids=position_ids, slot_mapping=slot_mapping,
        cu_seqlens_q=cu_seqlens, max_seqlen_q=max_seqlen,
        cu_seqlens_k=cu_seqlens, max_seqlen_k=max_seqlen,   # q == k for pure prefill
        cache_seqlens=lengths,                              # post-prefill KV depth per req
    )

def build_decode_metadata(past_lens, cache: KVCache, cache_len):
    B = past_lens.numel()
    rows = torch.arange(B, device=past_lens.device)
    kv_lens = past_lens.to(torch.int32) + 1                       # post-write depth
    cu_seqlens_q = torch.arange(B + 1, device=past_lens.device, dtype=torch.int32)   # cumsum([1, 1, ...]) -> [0,1,...,B]
    cu_seqlens_k = torch.zeros(B + 1, device=past_lens.device, dtype=torch.int32)
    cu_seqlens_k[1:] = torch.cumsum(kv_lens, 0)
    slot_mapping=rows * cache_len + past_lens       # write at depth past_len
    return AttentionMetadata(
        is_prefill=False, cache=cache,
        position_ids=past_lens,                          # abs pos of the new token
        slot_mapping=slot_mapping,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=1,
        cu_seqlens_k=cu_seqlens_k, max_seqlen_k=int(kv_lens.max()),
        cache_seqlens=kv_lens,                           # post-write depth (see note)
    )

def sdpa_one_seq(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # LAYOUT: padded batch only
    # q, k, v: (B, H, S, D)
    n_rep = q.size(1) // k.size(1)
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    q_len, kv_len = q.size(-2), k.size(-2)
    if q_len == kv_len:
        # full prefill: is_causal top-left == bottom-right here; take the fast path
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if q_len == 1:
        # decode: single new token sees all cached keys, no mask needed
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)
    # chunked prefill: must use bottom-right alignment, NOT is_causal (top-left)
    mask = _bottom_right_causal_bias(q_len, kv_len, q.device, q.dtype)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

def sdpa_varlen_fallback(
    q: Tensor, k: Tensor, v: Tensor,          # (total_tokens, H, D), packed
    meta: AttentionMetadata,
) -> Tensor:
    out = torch.empty_like(q)
    num_seqs = meta.cu_seqlens_q.numel() - 1
    for i in range(num_seqs):
        qs, qe = meta.cu_seqlens_q[i], meta.cu_seqlens_q[i + 1]
        ks, ke = meta.cu_seqlens_k[i], meta.cu_seqlens_k[i + 1]
        # slice one sequence, add batch dim, move to (1, H, s_i, D) for SDPA
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)   # (1, H, q_i, D)
        ki = k[ks:ke].transpose(0, 1).unsqueeze(0)   # (1, H, k_i, D)
        vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
        oi = sdpa_one_seq(qi, ki, vi)               # per-seq phase dispatch below
        out[qs:qe] = oi.squeeze(0).transpose(0, 1)   # back to (q_i, H, D)
    return out

def sdpa_from_cache(
    q: Tensor,                 # [T, Hq, D] packed queries, post-RoPE
    k_cache: Tensor,           # [max_seqs, cache_len, Hkv, D]
    v_cache: Tensor,
    meta: AttentionMetadata,
) -> Tensor:
    # Reference path. Reads KV back from the cache in BOTH phases, so that
    # scatter_to_kv_cache / slot_mapping stay under test on the CPU box —
    # they have no HF oracle of their own.
    out = torch.empty_like(q)
    cu_q = meta.cu_seqlens_q.tolist()
    kv_lens = meta.cache_seqlens.tolist()

    for i, kv_len in enumerate(kv_lens):
        qs, qe = cu_q[i], cu_q[i + 1]
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)                # [1, Hq,  q_i,    D]
        ki = k_cache[i, :kv_len].transpose(0, 1).unsqueeze(0)   # [1, Hkv, kv_len, D]
        vi = v_cache[i, :kv_len].transpose(0, 1).unsqueeze(0)
        oi = sdpa_one_seq(qi, ki, vi)                             # [1, Hq, q_i, D]
        out[qs:qe] = oi.squeeze(0).transpose(0, 1)                # [q_i, Hq, D]
    return out

def scatter_to_kv_cache(k_cache, v_cache, k, v, slot_mapping):
    # k_cache, v_cache: [max_seqs, cache_len, n_kv_heads, head_dim]
    # k, v            : [total_tokens, n_kv_heads, head_dim] packed; k POST-RoPE, v raw
    # slot_mapping    : [total_tokens] int64, = row*cache_len + position
    H, D = k.shape[1], k.shape[2]
    k_flat = k_cache.view(-1, H, D)          # [max_seqs*cache_len, H, D] view, no copy
    v_flat = v_cache.view(-1, H, D)
    k_flat.index_copy_(0, slot_mapping, k.to(k_flat.dtype))
    v_flat.index_copy_(0, slot_mapping, v.to(v_flat.dtype))

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.rope = init_rope(cfg)

        self.num_query_heads = cfg.num_attention_heads
        self.num_key_value_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.attn_dim = cfg.hidden_size

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def _attn(self, q, k, v, k_cache, v_cache, meta):
        if not HAS_FLASH_ATTN:
            if meta.cache is None:
                return sdpa_varlen_fallback(q, k, v, meta)
            else:
                return sdpa_from_cache(q, k_cache, v_cache, meta)

        if meta.is_prefill:
            # full prefill: packed k/v IS the complete KV (cu_seqlens_q == cu_seqlens_k).
            # The cache is write-only here; varlen cannot express the slab stride anyway.
            return flash_attn_varlen_func(
                q, k, v,
                meta.cu_seqlens_q, meta.cu_seqlens_k,
                meta.max_seqlen_q, meta.max_seqlen_k,
                causal=True,
            )                                        # [T, Hq, D]

        # decode: the new token is already in the cache (scatter ran above), so k=v=None.
        # cache_batch_idx maps batch slot -> physical cache row; required once B != max_seqs.
        return flash_attn_with_kvcache(
            q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=meta.cache_seqlens,
            cache_batch_idx=meta.cache_batch_idx,
            causal=True,
        ).squeeze(1)                                 # [B, 1, Hq, D] -> [B, Hq, D]

    def forward(self, hidden_states: Tensor, meta: AttentionMetadata) -> Tensor:
        # [T, hidden_size]
        T, _ = hidden_states.size()

        # projection
        query_states = self.q_proj(hidden_states).view(T, self.num_query_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(T, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(T, self.num_key_value_heads, self.head_dim)

        # rope
        query_states, key_states = self.rope.forward(query_states, key_states, meta.position_ids)

        # debug
        if meta.debug_k_list is not None and meta.debug_v_list is not None:
            meta.debug_k_list.append(key_states)
            meta.debug_v_list.append(value_states)

        #kv cache
        k_cache = v_cache = None
        if meta.cache is not None:
            k_cache, v_cache = meta.cache.data[self.layer_index]
            scatter_to_kv_cache(k_cache, v_cache, key_states, value_states, meta.slot_mapping)

        attn_out = self._attn(query_states, key_states, value_states, k_cache, v_cache, meta)

        return self.o_proj(attn_out.reshape(T, self.attn_dim))


