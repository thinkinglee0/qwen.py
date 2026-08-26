import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import logging
import itertools
import numpy as np

from qwen.config import ModelConfig
from qwen.cache import KVCacheData
from qwen.scheduler import SchedulerOutput
from qwen.rope import BaseRoPE


# attention backend selection — resolved once at import
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache  # for CUDA
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
    cache: KVCacheData

    # query side
    cu_seqlens_q: Tensor   # (num_seqs + 1,) prefix-sum of query lengths
    max_seqlen_q: int

    # lens of kv cache
    cu_seqlens_k: Tensor
    max_seqlen_k: int
    cache_seqlens: Tensor
    block_table:  Tensor    # [num_seqs, max_blocks]
    # kv for SDPA
    # block_tables: list[list[int]]   # [num_seqs, num_blocks]
    # k_lens:       list[int]         # 

    position_ids: Tensor    # rope
    slot_mapping: Tensor    # scatter q/v projections to kv cache

    # debug cache issue
    debug_k_list: list[Tensor] | None = None
    debug_v_list: list[Tensor] | None = None

def build_block_table(tables, device):
    max_blocks = max(len(t) for t in tables)
    bt_np = np.zeros((len(tables), max_blocks), dtype=np.int32)
    for i, t in enumerate(tables):
        bt_np[i, :len(t)] = t
    bt = torch.from_numpy(bt_np).to(device, non_blocking=True)   # single H2D

    # one H2D operation per request
    # bt = torch.zeros(len(tables), max_blocks, dtype=torch.int32, device=device)
    # for i, t in enumerate(tables):
    #     bt[i, :len(t)] = torch.tensor(t, dtype=torch.int32)       # H2D

    return bt   # [num_seqs, max_blocks];  padding entries never read, truncated by seqused_k

def build_attn_metadata(sch_out: SchedulerOutput, cache_data: KVCacheData, device) -> tuple[Tensor, AttentionMetadata]:
    # packing
    packed_id_list: list[int] = []
    lens: list[int] = []
    cache_lens: list[int] = []
    position_id_lst: list[int] = []
    slots: list[int] = []
    for req in sch_out.reqs:
        s_info = sch_out.scheduled[req.request_id]
        lens.append(s_info.want)
        slots.extend(s_info.slots)
        packed_id_list.extend(req.get_existing_ids(s_info.want))    # compatible for recompute

        start = req.num_computed_tokens
        end = start + s_info.want
        cache_lens.append(end)
        position_id_lst.extend(range(start, end))

    # preparation
    packed_ids = torch.tensor(packed_id_list, device=device, dtype=torch.int32)      # [T]

    max_seqlen_q = max(lens)
    cu_seqlens_q = torch.tensor(
        list(itertools.accumulate(lens, initial=0)), device=device, dtype=torch.int32
    )

    cache_seqlens = torch.tensor(cache_lens, device=device, dtype=torch.int32)
    max_seqlen_k = max(cache_lens)
    cu_seqlens_k = torch.tensor(
        list(itertools.accumulate(cache_lens, initial=0)), device=device, dtype=torch.int32
    )

    position_ids = torch.tensor(position_id_lst, device=device, dtype=torch.int32)
    slot_mapping = torch.tensor(slots, device=device, dtype=torch.int64)

    block_table = build_block_table(sch_out.block_tables, device)

    return packed_ids, AttentionMetadata(cache=cache_data,
                             cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
                             cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k,
                             cache_seqlens=cache_seqlens, block_table=block_table,
                             position_ids=position_ids, slot_mapping=slot_mapping)

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

def gather_kv_cache(table: list[int], kv_len: int, k_cache: torch.Tensor, v_cache: torch.Tensor):
    _, block_size, H, D = k_cache.size()

    k = torch.empty(kv_len, H, D, dtype=k_cache.dtype, device=k_cache.device)
    v = torch.empty(kv_len, H, D, dtype=k_cache.dtype, device=k_cache.device)

    # # copy token by token
    # for p in range(kv_len):
    #     logic_block_id = p // block_size
    #     physical_block_id = table[logic_block_id]
    #     offset = p % block_size
    #     k[p] = k_cache[physical_block_id, offset]
    #     v[p] = v_cache[physical_block_id, offset]

    # copy block by block. full block
    n_blocks = kv_len // block_size
    for idx in range(n_blocks):
        physical_block = table[idx]
        k[idx*block_size:(idx+1)*block_size, :] = k_cache[physical_block, :]
        v[idx*block_size:(idx+1)*block_size, :] = v_cache[physical_block, :]

    # partial block, that is, last block
    remainder = kv_len % block_size
    if remainder:
        idx = n_blocks
        physical_block = table[idx]
        k[idx*block_size:kv_len, :] = k_cache[physical_block, :remainder, :]
        v[idx*block_size:kv_len, :] = v_cache[physical_block, :remainder, :]

    return k, v     # [k_len, H, D]

def sdpa_from_cache(
    q: Tensor,                 # [T, Hq, D] packed queries, post-RoPE
    k_cache: Tensor,           # [num_blocks, block_size, Hkv, D]
    v_cache: Tensor,
    meta: AttentionMetadata,
) -> Tensor:
    # Reference path. Reads KV back from the cache in BOTH phases, so that
    # scatter_to_kv_cache / slot_mapping stay under test on the CPU box —
    # they have no HF oracle of their own.
    out = torch.empty_like(q)
    cu_q = meta.cu_seqlens_q.tolist()
    kv_lens = meta.cache_seqlens.tolist()
    block_tables = meta.block_table.tolist()

    for i, kv_len in enumerate(kv_lens):
        qs, qe = cu_q[i], cu_q[i + 1]
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)                # [1, Hq,  q_i,    D]
        ki, vi = gather_kv_cache(block_tables[i], kv_len, k_cache, v_cache)
        ki = ki.transpose(0, 1).unsqueeze(0)   # [1, Hkv, kv_len, D]
        vi = vi.transpose(0, 1).unsqueeze(0)
        oi = sdpa_one_seq(qi, ki, vi)                             # [1, Hq, q_i, D]
        out[qs:qe] = oi.squeeze(0).transpose(0, 1)                # [q_i, Hq, D]
    return out

def scatter_to_kv_cache(k_cache, v_cache, k, v, slot_mapping):
    # k_cache, v_cache: [num_blocks, block_size, n_kv_heads, head_dim]
    # k, v            : [total_tokens, n_kv_heads, head_dim] packed; k post-RoPE, v raw
    # slot_mapping    : [total_tokens] int64, = block_id*block_size + offset
    H, D = k.shape[1], k.shape[2]
    k_flat = k_cache.view(-1, H, D)          # [T, H, D] view, no copy
    v_flat = v_cache.view(-1, H, D)
    k_flat.index_copy_(0, slot_mapping, k.to(k_flat.dtype))
    v_flat.index_copy_(0, slot_mapping, v.to(v_flat.dtype))

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_index: int, rope: BaseRoPE):
        super().__init__()
        self.layer_index = layer_index
        self.rope = rope

        self.num_query_heads = cfg.num_attention_heads
        self.num_key_value_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.attn_dim = cfg.hidden_size

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def _attn(self, q, k_cache, v_cache, meta: AttentionMetadata):
        if not HAS_FLASH_ATTN:
            return sdpa_from_cache(q, k_cache, v_cache, meta)

        assert flash_attn_varlen_func, "flash_attn not installed"
        return flash_attn_varlen_func(
            q, k_cache, v_cache,
            cu_seqlens_q=meta.cu_seqlens_q,     # [num_reqs+1]  query boundary
            max_seqlen_q=meta.max_seqlen_q,
            cu_seqlens_k=meta.cu_seqlens_k,     # [num_reqs+1]  query boundary
            max_seqlen_k=meta.max_seqlen_k,
            block_table=meta.block_table,       # [num_reqs, max_num_blocks_per_req]
            causal=True,
        )                                       # [T, Hq, D]

    def forward(self, hidden_states: Tensor, meta: AttentionMetadata) -> Tensor:
        # [T, hidden_size]
        T, _ = hidden_states.size()

        # projection [T, H, D]
        query_states = self.q_proj(hidden_states).view(T, self.num_query_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(T, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(T, self.num_key_value_heads, self.head_dim)

        # rope
        query_states, key_states = self.rope.forward(query_states, key_states, meta.position_ids)

        # debug
        if meta.debug_k_list is not None and meta.debug_v_list is not None:
            meta.debug_k_list.append(key_states)
            meta.debug_v_list.append(value_states)

        # update kv cache
        k_cache, v_cache = meta.cache.k_caches[self.layer_index], meta.cache.v_caches[self.layer_index]
        scatter_to_kv_cache(k_cache, v_cache, key_states, value_states, meta.slot_mapping)

        attn_out = self._attn(query_states, k_cache, v_cache, meta)

        return self.o_proj(attn_out.reshape(T, self.attn_dim))


