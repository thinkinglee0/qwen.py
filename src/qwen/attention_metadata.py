import torch
from torch import Tensor
from dataclasses import dataclass

from qwen.rope import BaseRoPE
from qwen.metrics import SchedulerStepMetrics, timed
from qwen.cache import KVCacheData


@dataclass
class AttentionMetadata:

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

    slot_mapping: Tensor    # scatter q/v projections to kv cache
    position_ids: Tensor    # rope
    rope: BaseRoPE | None = None
    cos_sin: tuple[Tensor, Tensor] | None = None   # optional pre-gathered cos/sin for rope

    # metrics
    step_metrics_lst: list[SchedulerStepMetrics] | None = None

    cache: KVCacheData | None = None

    # debug cache issue
    debug_k_list: list[Tensor] | None = None
    debug_v_list: list[Tensor] | None = None

    def layer_metrics(self, layer_index: int) -> SchedulerStepMetrics | None:
        """Per-layer metrics slot, or None when metrics collection is off."""
        return self.step_metrics_lst[layer_index] if self.step_metrics_lst is not None else None