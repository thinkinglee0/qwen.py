from collections import deque
import logging
import dataclasses
from dataclasses import dataclass
import asyncio
import time
from datetime import datetime
from pathlib import Path
import copy
import orjson
import uuid
import torch
import numpy as np
from itertools import chain

from qwen.utils import round_floats
from qwen.sampling import SamplingTensors, SamplingParams, SamplingParamTable
from qwen.config import ModelConfig
from qwen.metrics import analyze_metrics, RequestMetrics, SchedulerMetrics, SchedulerStepMetrics, timed
from qwen.cache import KVCache, cdiv
from qwen.slot import RequestSlotPool
from qwen.constants import DEFAULT_MAX_NEW_TOKEN
from qwen.attention_metadata import AttentionMetadata
from qwen.rope import BaseRoPE
from qwen.cache import KVCacheData

logger = logging.getLogger(__name__)


class RequestBuffer:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

        # row 0: slot index, 1: want
        self.slot_idx_device = torch.empty(cfg.max_num_seqs, dtype=torch.int64, device=cfg.device)
        self.slot_idx_stage = torch.empty(cfg.max_num_seqs, dtype=torch.int64, pin_memory=True) \
            if torch.cuda.is_available() else torch.empty(cfg.max_num_seqs, dtype=torch.int64)
        
        self.want_device = torch.empty(cfg.max_num_seqs, dtype=torch.int32, device=cfg.device)
        self.want_stage = torch.empty(cfg.max_num_seqs, dtype=torch.int32, pin_memory=True) \
            if torch.cuda.is_available() else torch.empty(cfg.max_num_seqs, dtype=torch.int32)

        # needs_sample
        self.needs_sample_device = torch.empty(cfg.max_num_seqs, dtype=torch.bool, device=cfg.device)
        self.needs_sample_stage = torch.empty(cfg.max_num_seqs, dtype=torch.bool, pin_memory=True) \
            if torch.cuda.is_available() else torch.empty(cfg.max_num_seqs, dtype=torch.bool)

        # cache_slots
        self.cache_slot_device = torch.empty(cfg.max_num_batched_tokens, dtype=torch.int64, device=cfg.device)
        self.cache_slot_stage = torch.empty(cfg.max_num_batched_tokens, dtype=torch.int64, pin_memory=True) \
            if torch.cuda.is_available() else torch.empty(cfg.max_num_batched_tokens, dtype=torch.int64)

        # block table
        self.max_block_num_per_req = cdiv(cfg.max_model_len, cfg.block_size)
        self.block_table_device = torch.empty(cfg.max_num_seqs, self.max_block_num_per_req, dtype=torch.int32, device=cfg.device)
        self.block_table_stage = torch.empty(cfg.max_num_seqs, self.max_block_num_per_req, dtype=torch.int32, pin_memory=True) \
            if torch.cuda.is_available() else torch.empty(cfg.max_num_seqs, self.max_block_num_per_req, dtype=torch.int32)

        # view of numpy
        self.slot_idx_np = self.slot_idx_stage.numpy()
        self.want_np = self.want_stage.numpy()
        self.needs_sample_np = self.needs_sample_stage.numpy()
        self.cache_slot_np = self.cache_slot_stage.numpy()
        self.block_table_np = self.block_table_stage.numpy()

        self._blk_cols = np.arange(self.max_block_num_per_req)  # costant variable

    def _build_block_table(self, block_tables: list[list[int]]) -> torch.Tensor:
        bsz = len(block_tables)
        # lens = np.fromiter((len(r) for r in block_tables), dtype=np.int64, count=bsz)
        lens = np.fromiter(map(len, block_tables), dtype=np.int64, count=bsz)   # map more efficient than for loop

        max_blocks = int(lens.max())
        assert max_blocks <= self.max_block_num_per_req

        # optimization by numpy
        mask = self._blk_cols < lens[:, None]
        flat = list(chain.from_iterable(block_tables))  # [x for r in block_tables for x in r]

        view = self.block_table_np[:bsz]        # basic slice -> view, writes go through
        view[mask] = flat

        # oracle
        # for i, bt in enumerate(block_tables):
        #     for j, b in enumerate(bt):
        #         self.block_table_stage[i, j] = b

        self.block_table_device[:bsz, :max_blocks].copy_(self.block_table_stage[:bsz, :max_blocks], non_blocking=True)

        return self.block_table_device[:bsz, :max_blocks]

    def set(self, reqs: list["ModelRequest"], s_infos: list["ScheduledInfo"],
            block_tables: list[list[int]],
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = len(reqs)

        self.slot_idx_np[:bsz] = [r.slot for r in reqs]
        self.want_np[:bsz]     = [s.want for s in s_infos]
        self.needs_sample_np[:bsz] = [r.projected_is_decoding or
                                (r.num_scheduled_tokens + s.want == r._num_projected_prefill_tokens)
                                for r, s in zip(reqs, s_infos)]
        cache_slot_flat = list(chain.from_iterable(s.cache_slots for s in s_infos))
        cache_slot_cnt = len(cache_slot_flat)
        self.cache_slot_np[:cache_slot_cnt] = cache_slot_flat

        # async copy
        self.slot_idx_device[:bsz].copy_(self.slot_idx_stage[:bsz], non_blocking=True)
        self.want_device[:bsz].copy_(self.want_stage[:bsz], non_blocking=True)
        self.needs_sample_device[:bsz].copy_(self.needs_sample_stage[:bsz], non_blocking=True)
        self.cache_slot_device[:cache_slot_cnt].copy_(self.cache_slot_stage[:cache_slot_cnt], non_blocking=True)

        # block table
        block_table = self._build_block_table(block_tables=block_tables)

        return self.slot_idx_device[:bsz], self.slot_idx_stage[:bsz], \
            self.want_device[:bsz], self.want_stage[:bsz], \
            self.needs_sample_device[:bsz], self.needs_sample_stage[:bsz], \
            self.cache_slot_device[:cache_slot_cnt], block_table

class ModelRequest:
    def __init__(self, config: ModelConfig, loop, input_ids: list[int], request_id: str | None=None, sampling: SamplingParams | None = None, max_new_tokens: int=DEFAULT_MAX_NEW_TOKEN):
        self.request_id = request_id if request_id is not None else str(uuid.uuid4())
        self.input_ids = input_ids
        self.input_len = len(input_ids)
        self.sampling = sampling
        self.max_new_tokens = max(1, min(config.max_model_len-len(input_ids), max_new_tokens))
        self.loop = loop
        self.token_queue: asyncio.Queue[int | None | Exception] = asyncio.Queue()
        self.slot: int | None = None

        if self.sampling is not None:
            self.sampling.validate(config=config)

        # for backoff
        self.is_preempted: bool = False
        self.preempt_count: int = 0
        self.not_before_step: int = 0     # earliest scheduler step at which re-admission is allowed
        # decrease preempt_count when num_computed_tokens>num_computed_tokens_at_preempt
        self.num_computed_tokens_at_preempt:int = 0
        # number of in-flight steps, for _pick_waiting, supports multiple in-flight steps, but only single in-flight step is implemented.
        self.num_in_flight: int = 0

        # metrics
        self.metrics = RequestMetrics(arrival_time=time.perf_counter(), num_input_token=len(self.input_ids))

        # intermediate actual states
        self.output_ids: list[int] = []
        self.num_computed_tokens: int = 0
        self.finished: bool = False
        self.committed: bool = False

        # states projected by the scheduler: what the actual states above will become
        # once the in-flight step lands, so the next step can be planned without waiting.
        # NOTE: update before SchedulerOutput.add_sampled_tokens triggered by next_tokens' arrival at host.
        self.num_scheduled_tokens: int = 0   # the number of tokens the scheduler has scheduled
        self.projected_is_decoding: bool = False
        self.projected_finished: bool = False

    # keep output_ids for consistency from user's perspective.
    # only drop states about kv cache and queuing
    def reset_projected_states_on_preemption(self) -> None:
        self.num_scheduled_tokens = 0    # for kv cache
        self.projected_is_decoding = False        # phase

        # record this preemption
        self.is_preempted = True
        self.preempt_count += 1
        self.num_computed_tokens_at_preempt = self.num_computed_tokens

    # called by SchedulerOutput.add_sampled_tokens_on_host when next_tokens of a batch arrives at host from device.
    def reset_actual_states_on_preemption(self) -> None:
        self.num_computed_tokens = 0    # for kv cache

    def get_existing_ids(self, want: int) -> list[int]:
        assert 0 < want <= len(self.input_ids) + len(self.output_ids)

        start = self.num_computed_tokens
        end = start + want

        # oracle
        # return (self.input_ids + self.output_ids)[start:end]

        n_prompt = len(self.input_ids)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"request_id: {self.request_id}, start: {start}, end: {end}, n_prompt: {n_prompt}, n_out: {len(self.output_ids)}")
        if end <= n_prompt:                       # fully inside prompt (normal prefill)
            return self.input_ids[start:end]
        if start >= n_prompt:                     # fully inside output (decode / replay)
            s = start - n_prompt
            return self.output_ids[s:s + want]
        return self.input_ids[start:] + self.output_ids[: end - n_prompt]

    @property
    def _num_projected_prefill_tokens(self) -> int:
        if not self.projected_is_decoding:
            return len(self.input_ids) + len(self.output_ids)
        else:
            return len(self.input_ids)

    @property
    def num_projected_prompt_remaining(self) -> int:  # compatible with requests re-computing from scratch after evicted from decoding
        return max(0, self._num_projected_prefill_tokens - self.num_scheduled_tokens)

    # return 1 if this request is finished because the output tokens exceed max_new_tokens, meaning that the request is truncated
    def add_sampled_token(self, tok: int, eos_token_id_set, now) -> int:
        num_truncated: int = 0
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"add_sampled_token, {self.request_id}, tok: {tok}, projected_finished: {self.projected_finished}, "
                    f"out_len: {len(self.output_ids)}, max_new_tokens: {self.max_new_tokens}")
        if tok in eos_token_id_set \
            or len(self.output_ids)+1 >= self.max_new_tokens:
            self.finished = True

            if tok in eos_token_id_set:
                self.projected_finished = True          # update projected_finished synchronously
            else:
                assert self.projected_finished == True  # should be flipped in advance in this branch.
                num_truncated += 1

            # self.io_token_ids, do not append when finished
            self.output_ids.append(tok) if tok not in eos_token_id_set else None
            self.metrics.report(now)     # regardless of EOS or not

            if self.loop is not None:
                if tok not in eos_token_id_set:
                    self.loop.call_soon_threadsafe(self.token_queue.put_nowait, tok)
                self.loop.call_soon_threadsafe(self.token_queue.put_nowait, None)  # sentinel = stream end
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"SchedulerOutput: req_id: {self.request_id}, finished, len: {len(self.output_ids)}, sampled token={tok}")
        else:
            self.output_ids.append(tok)
            self.metrics.report(now)

            if self.loop is not None:
                self.loop.call_soon_threadsafe(self.token_queue.put_nowait, tok)  

        return num_truncated

@dataclass
class ScheduledInfo:
    want: int
    cache_slots: list[int]

    def __post_init__(self):
        assert self.want == len(self.cache_slots)

class TokenIdTable:
    def __init__(self, max_num_seqs: int, max_model_len: int, device: torch.device):
        '''periodically async-copy output token and output len from device to host'''
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len

        # shape [B, L], all token ids including prompt and output tokens per row
        self.tok_id_device = torch.empty(max_num_seqs, max_model_len, dtype=torch.int64, device=device)
        self.tok_id_stage = torch.empty(max_num_seqs, max_model_len, dtype=torch.int64, pin_memory=True) \
            if torch.cuda.is_available() else torch.empty(max_num_seqs, max_model_len, dtype=torch.int64)

        # row 0: input length, 1: output length, 2: numer of computed tokens
        tok_len_device = torch.empty(3, max_num_seqs, dtype=torch.int32, device=device)
        (self.in_len_device, self.out_len_device, self.num_computed_tok_device) = tok_len_device.unbind() # shape [B]
        # do not act as stages on DMA to out_len_device/num_computed_tok_device, so its device set to 'cpu'
        (self.in_len_host, self.projected_out_len_host, self.num_scheduled_tok_host) = \
            torch.empty(3, max_num_seqs, dtype=torch.int32, device=torch.device("cpu")).unbind()

        # double buffer
        # only host-side buffer, for transfer next tokens from device to host. top-bsz elements used.
        next_tok_stage_double_buffer = torch.empty(2, max_num_seqs, dtype=torch.int64, pin_memory=True) \
            if torch.cuda.is_available() else torch.empty(2, max_num_seqs, dtype=torch.int64)
        self.next_tok_stage_bufs = next_tok_stage_double_buffer.unbind() # tuple
        self.dth_buf: int = 0

    def add_req(self, req: "ModelRequest"):
        assert req.slot is not None

        slot = req.slot
        in_len = req.input_len
        out_len = len(req.output_ids)

        self.in_len_host[slot] = in_len
        self.projected_out_len_host[slot] = out_len
        self.num_scheduled_tok_host[slot] = 0

        # WARNING: these three MUST NOT be fed by a non_blocking H2D copy out of the pinned
        # host cells above. The DMA is queued behind a whole step of kernels, while
        # update_projected_state_in_advance() mutates num_computed_tok_host later in this very
        # step -- the copy would then land `want` instead of 0, and the device counter ends up
        # at 2*want. Fill on the device instead: no host source, no race. Admissions are rare
        # compared to steps, so the three tiny kernels do not matter.
        # self.in_len_device[slot].copy_(self.in_len_host[slot], non_blocking=True)
        # self.out_len_device[slot].copy_(self.out_len_host[slot], non_blocking=True)
        # self.num_computed_tok_device[slot].copy_(self.num_computed_tok_host[slot], non_blocking=True)
        self.in_len_device[slot].fill_(in_len)
        self.out_len_device[slot].fill_(out_len)
        self.num_computed_tok_device[slot].zero_()

        for i, tok in enumerate(req.input_ids):
            self.tok_id_stage[slot, i] = tok
        for i, tok in enumerate(req.output_ids):        # for recompute
            self.tok_id_stage[slot, i+in_len] = tok
        self.tok_id_device[slot, :in_len+out_len].copy_(self.tok_id_stage[slot, :in_len+out_len], non_blocking=True)

    # scatter next tokens to tok_id_device, then copy them from device to host asynchronously.
    def add_sampled_tokens_on_device(
            self,
            slot_idx: torch.Tensor,         # [B]
            needs_sample: torch.Tensor,     # [B]
            next_tokens: torch.Tensor,      # [B]
            want: torch.Tensor,             # [B]
            step_metrics: SchedulerStepMetrics):
        bsz = slot_idx.numel()
        assert needs_sample.numel() == bsz and next_tokens.numel() == bsz and want.numel() == bsz

        # write position of each row
        pos = self.in_len_device.index_select(0, slot_idx) + self.out_len_device.index_select(0, slot_idx)
        pos.clamp_(max=self.max_model_len - 1)

        # flat scatter
        flat_idx = slot_idx * self.max_model_len + pos
        tok_flat = self.tok_id_device.view(-1)
        keep = tok_flat.gather(0, flat_idx)
        tok_flat.scatter_(0, flat_idx, torch.where(needs_sample, next_tokens, keep))

        # only the rows that actually took a token advance their output length
        self.out_len_device.index_add_(0, slot_idx, needs_sample.to(self.out_len_device.dtype))

        self.num_computed_tok_device.index_add_(0, slot_idx, want)

        # async copy next tokens from device to host, then add them to ModelRequest.token_queue when the async copy finishes.
        buf = self.dth_buf
        with timed(step_metrics, "dth"):
            self.next_tok_stage_bufs[self.dth_buf][:bsz].copy_(next_tokens, non_blocking=True)
        self.dth_buf ^= 1   # flip
        return buf

    def update_projected_state_in_advance(self, slot_idx_stage: torch.Tensor, want_stage: torch.Tensor, needs_sample_stage_snapshot: torch.Tensor):
        self.num_scheduled_tok_host.index_add_(0, slot_idx_stage, want_stage)
        self.projected_out_len_host.index_add_(0, slot_idx_stage, needs_sample_stage_snapshot.int())

    def gather_flat_pending_tok(self, slot_idx: torch.Tensor, slot_idx_stage: torch.Tensor, want: torch.Tensor, want_stage: torch.Tensor, num_tokens: int
                                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        bsz = slot_idx.numel()
        assert want.numel() == bsz and num_tokens >= bsz

        # element-wise comparison
        exceeds = (self.num_scheduled_tok_host[slot_idx_stage] + want_stage) > \
            (self.in_len_host[slot_idx_stage] + self.projected_out_len_host[slot_idx_stage])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"num_scheduled_tok_host: {self.num_scheduled_tok_host[slot_idx_stage]}, want_stage: {want_stage}, "
                    f"in_len_host: {self.in_len_host[slot_idx_stage]}, projected_out_len_host: {self.projected_out_len_host[slot_idx_stage]}")
        if exceeds.any():
            raise ValueError("want overflow, scheduler must malfunction")

        cache_seqlens = want + self.num_computed_tok_device[slot_idx]   # [B]
        _cu_seqlens_k = torch.cumsum(cache_seqlens, 0, dtype=torch.int32)                  # [B]
        cu_seqlens_k = torch.cat([_cu_seqlens_k.new_zeros(1), _cu_seqlens_k])   # [B+1], begin with zero

        # max_seqlen_k in host-side
        cache_seqlens_host = want_stage + self.num_scheduled_tok_host[slot_idx_stage]
        max_seqlen_k = int(torch.max(cache_seqlens_host).item())

        cu_want = torch.cumsum(want, 0, dtype=torch.int32)      # [B]
        cu_seqlens_q = torch.cat([cu_want.new_zeros(1), cu_want])       # [B+1], begin with zero
        t = torch.arange(num_tokens, device=self.tok_id_device.device, dtype=cu_want.dtype)    # [T]
        batch_idx = torch.searchsorted(cu_want, t, right=True)          # [T], in [0, B)
        row = slot_idx[batch_idx].to(torch.int64)       # [T], the slot each token belongs to
        if num_tokens == bsz:
            # all reqs are in decoding, row==slot_idx
            position_ids = self.num_computed_tok_device[slot_idx]
        else:
            # read position of each row
            # NOTE: num_computed_tok_device is indexed by SLOT, not by batch position,
            # so index it with row (== slot_idx[batch_idx]) instead of batch_idx.
            offset = cu_want - want              # [B]
            position_ids = self.num_computed_tok_device[row] + (t - offset[batch_idx]) # [T], in [0, max_model_len)

        # position_ids -> flat_position -> pending token ids
        flat_position = row * self.max_model_len + position_ids     # [T]
        flat_pending_tok = self.tok_id_device.view(-1).index_select(0, flat_position)    # [T]
        return flat_pending_tok, position_ids, cu_seqlens_q, cache_seqlens, cu_seqlens_k, max_seqlen_k

    def bin_count_and_mask(self,
        slot_idx: torch.Tensor,      # [B], GPU
        vocab_size: int,             # host int (logits dim, e.g. 151936 for Qwen2.5)
    ):
        B, V = slot_idx.shape[0], vocab_size
        L = self.max_model_len
        dev = self.tok_id_device.device

        # [B, L] history rows; scatter indices must be int64
        rows = self.tok_id_device[:, :L].index_select(0, slot_idx).to(torch.int64)     # [B, L]
        p_len = self.in_len_device.index_select(0, slot_idx)[:, None]                  # [B, 1]
        s_len = p_len + self.out_len_device.index_select(0, slot_idx)[:, None]         # [B, 1]
        j = torch.arange(L, device=dev)[None, :]                    # [1, L]
        is_prompt = j < p_len                                       # [B, L]
        is_valid = j < s_len                                        # [B, L], excludes padding region
        is_output = is_valid & ~is_prompt                           # [B, L]

        output_counts = torch.zeros(B, V, dtype=torch.int32, device=dev)
        # output_counts.scatter_add_(1, torch.where(is_output, rows, 0), is_output.to(torch.int32)) # all dummy elements with ~is_output add to index 0
        # optimization of the line above: spread the dummy additions away from index 0
        idx = torch.where(is_output, rows, j.expand_as(rows))      # requires L <= V; else use j % V
        output_counts.scatter_add_(1, idx, is_output.to(torch.int32))

        # both masks in one buffer: [prompt V | output V | dump 16]; one memset + one scatter
        # row stride 2V+16 keeps rows 16B-aligned when V % 16 == 0 (vectorized downstream loads)
        DUMP = 2 * V
        masks = torch.zeros(B, 2 * V + 16, dtype=torch.bool, device=dev)
        # keep in mind that the tok id in rows is the index of vocabulary
        # rows [B, L], is_output [B, L]
        out_offset = V * is_output
        vocab_idx = torch.where(is_valid, rows + out_offset, DUMP)  # [B, 2V+16], move the range of output tokens to the sencond V region by add V.
        masks.scatter_(1, vocab_idx, True)                          # racing writes of same value: safe

        return masks[:, :V], output_counts, masks[:, V : 2 * V]     # prompt_mask, output_counts, output_mask


class SchedulerOutput:
    def __init__(self, step_id: int, reqs: list[ModelRequest],
                 scheduled: dict[str, ScheduledInfo],
                 block_tables: list[list[int]],
                 config: ModelConfig,
                 scheduler: "Scheduler | None" = None,
                 step_metrics: SchedulerStepMetrics | None = None):
        assert len(reqs) > 0, "SchedulerOutput must have at least one request"
        self.reqs = reqs
        self.batch_size = len(reqs)
        self.s_infos = [s for s in scheduled.values()]
        self.block_tables = block_tables
        self.config = config
        self.scheduler = scheduler
        self.slot_idx: torch.Tensor | None = None
        self.slot_idx_stage_snapshot: torch.Tensor | None = None
        self.want: torch.Tensor | None = None
        self.want_stage_snapshot: torch.Tensor | None = None
        self.attn_meta: AttentionMetadata | None = None
        self.needs_sample_device: torch.Tensor | None = None
        # WARNING: do not save a view of a host-side tensor as below, because it will be updated in the next step.
        # self.needs_sample_stage: torch.Tensor | None = None
        # USING a native variable and a snapshot of it instead
        self.needs_sample_stage_snapshot: torch.Tensor | None = None
        self.needs_sample_lst: list[bool] | None = None
        self.dth_buf: int = 0

        # intermediate states for the batch
        self.output_ids: list[list[int]] = [req.output_ids for req in self.reqs]      # for repetition penalty and synchronous generation

        # metrics
        num_prefill, num_prefill_tokens, num_decode_tokens = 0, 0, 0
        for i, req in enumerate(self.reqs):
            if req.metrics.first_schedule_time is None:
                req.metrics.first_schedule_time = time.perf_counter()
            if not req.projected_is_decoding:
                # prefill
                num_prefill += 1
                num_prefill_tokens += self.s_infos[i].want
                req.metrics.num_prefill_chunk += 1
            else:
                # decode
                num_decode_tokens += 1

        self.step_metrics: SchedulerStepMetrics | None = step_metrics
        if scheduler is not None and self.step_metrics is not None:
            self.step_metrics.step_id = step_id
            self.step_metrics.bz = self.batch_size
            self.step_metrics.n_p = num_prefill
            self.step_metrics.n_p_tok = num_prefill_tokens
            self.step_metrics.n_d = num_decode_tokens
            self.step_metrics.run = len(scheduler.running)
            self.step_metrics.wait = len(scheduler.waiting)
            self.step_metrics.blk_used = scheduler.cache.pool.used()

    def build_sampling_tensors(self) -> SamplingTensors:
        assert self.scheduler is not None and self.scheduler.sampling_param_tab is not None
        assert self.slot_idx is not None and self.slot_idx_stage_snapshot is not None

        prompt_mask, output_counts, output_mask = self.scheduler.tok_id_tab.bin_count_and_mask(slot_idx=self.slot_idx, vocab_size=self.config.vocab_size)
        return SamplingTensors.from_table(
            sampling_param_tab=self.scheduler.sampling_param_tab,
            slot_idx=self.slot_idx,
            slot_idx_stage=self.slot_idx_stage_snapshot,
            prompt_mask=prompt_mask,
            output_counts=output_counts, output_mask=output_mask,)

    def incr_num_in_flight(self):
        for req in self.reqs:
            req.num_in_flight += 1

    def desc_num_in_flight(self):
        for req in self.reqs:
            req.num_in_flight -= 1
            assert req.num_in_flight >= 0

    def update_projected_state_in_advance(self):
        assert self.scheduler is not None and self.needs_sample_stage_snapshot is not None
        assert self.want_stage_snapshot is not None and self.slot_idx_stage_snapshot is not None
        self.scheduler.tok_id_tab.update_projected_state_in_advance(
            slot_idx_stage=self.slot_idx_stage_snapshot, want_stage=self.want_stage_snapshot,
            needs_sample_stage_snapshot=self.needs_sample_stage_snapshot)

        wants = self.want_stage_snapshot.tolist()
        for i, req in enumerate(self.reqs):
            assert not req.projected_finished

            # NOTE: use list instead of tensor to avoid the overhead of aten::item/select/as_strided
            # req.num_scheduled_tokens += int(self.want_stage_snapshot[i].item())
            req.num_scheduled_tokens += wants[i]

            assert self.needs_sample_lst is not None
            if not req.projected_is_decoding and self.needs_sample_lst[i]:
                req.projected_is_decoding = True
                if logger.isEnabledFor(logging.DEBUG):
                    assert self.step_metrics is not None
                    logger.debug(f"projected_is_decoding flips to true, step_id: {self.step_metrics.step_id}, {req.request_id}, "
                                 f"num_scheduled_tokens: {req.num_scheduled_tokens}, num_computed_tokens: {req.num_computed_tokens}")

            expected_out_len = max(0, req.num_scheduled_tokens - req.input_len + 1)    # todo
            assert expected_out_len <= req.max_new_tokens
            if expected_out_len == req.max_new_tokens:
                req.projected_finished = True
                if logger.isEnabledFor(logging.DEBUG):
                    assert self.step_metrics is not None
                    logger.debug(f"projected_finished flips to true, step_id: {self.step_metrics.step_id}, {req.request_id}, "
                                 f"num_scheduled_tokens: {req.num_scheduled_tokens}, num_computed_tokens: {req.num_computed_tokens}, "
                                 f"max_new_tokens: {req.max_new_tokens}, input_len: {req.input_len}")

    def build_attn_metadata(self, cache_data: KVCacheData | None = None, rope: BaseRoPE | None = None) -> tuple[torch.Tensor, AttentionMetadata]:
        assert self.scheduler is not None

        lens = [s.want for s in self.s_infos]
        num_tokens = sum(lens)
        max_seqlen_q = max(lens)
        self.slot_idx, slot_idx_stage, self.want, want_stage, self.needs_sample_device, needs_sample_stage, cache_slot, block_table = \
            self.scheduler.req_buf.set(reqs=self.reqs, s_infos=self.s_infos, block_tables=self.block_tables)
        self.needs_sample_stage_snapshot = needs_sample_stage.clone().detach()
        self.needs_sample_lst = needs_sample_stage.tolist()
        self.slot_idx_stage_snapshot = slot_idx_stage.clone().detach()
        self.want_stage_snapshot = want_stage.clone().detach()

        packed_ids, position_ids, cu_seqlens_q, cache_seqlens, cu_seqlens_k, max_seqlen_k = \
            self.scheduler.tok_id_tab.gather_flat_pending_tok(
                slot_idx=self.slot_idx,
                slot_idx_stage=slot_idx_stage,
                want=self.want, want_stage=want_stage,
                num_tokens=num_tokens)

        cos_sin = rope.gather_cos_sin(position_ids) if rope is not None and self.config.pre_gather_cos_sin else None

        # for each layer
        inner_step_metrics_lst = None
        if self.step_metrics is not None:
            inner_step_metrics_lst = [SchedulerStepMetrics() for _ in range(self.config.num_hidden_layers)]

        self.attn_meta = AttentionMetadata(cache=cache_data, rope=rope,
                                            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
                                            cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k,
                                            cache_seqlens=cache_seqlens, block_table=block_table,
                                            position_ids=position_ids, slot_mapping=cache_slot,
                                            cos_sin=cos_sin,
                                            step_metrics_lst=inner_step_metrics_lst,
                                            )
        return packed_ids, self.attn_meta

    # after transferring next_tokens from device to host
    def add_sampled_tokens_on_host(self) -> int:
        assert self.scheduler is not None and self.slot_idx is not None and self.step_metrics is not None
        next_tok_lst = self.scheduler.tok_id_tab.next_tok_stage_bufs[self.dth_buf][:self.batch_size].tolist()
        num_truncated: int = 0

        now = time.perf_counter()
        for i, (tok, req) in enumerate(zip(next_tok_lst, self.reqs)):
            if req.finished:    # discard due to EOS in the previous step, or due to error/abort
                continue

            s_info = self.s_infos[i]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"add_sampled_tokens_on_host, before adding, step_id: {self.step_metrics.step_id}, req_id: {req.request_id}, "
                             f"projected_is_decoding: {req.projected_is_decoding}, "
                             f"num_computed_tokens: {req.num_computed_tokens}, num_scheduled_tokens: {req.num_scheduled_tokens}, "
                             f"want: {s_info.want}, out_len: {len(req.output_ids)}, max: {req.max_new_tokens}, tok={tok}")

            req.num_computed_tokens += s_info.want

            # NOTE: must be handled before the needs_sample shortcut below, otherwise a victim that was
            # preempted while prefilling (needs_sample == False) would keep its stale num_computed_tokens.
            if req.is_preempted and req.num_in_flight == 1:
                # `req.num_in_flight` supports multiple in-flight stpes.
                # catch up with projected states which were set in advance only while the last in-flight step finished.
                req.reset_actual_states_on_preemption()
                if req.num_computed_tokens > req.num_computed_tokens_at_preempt:
                    req.preempt_count = max(0, req.preempt_count - 1)   # decrease, for consecutive jitters

                req.is_preempted = False    # the batch it lay in has landed, so it becomes selectable again.

            assert self.needs_sample_lst is not None
            if not bool(self.needs_sample_lst[i]):
                continue

            assert req.projected_is_decoding == True    # should be flipped in advance.

            # in decoding
            num_truncated += req.add_sampled_token(tok, self.config.eos_token_id_set, now)

        return num_truncated

class Scheduler:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.max_waiting = config.max_waiting
        self.long_prefill_token_threshold = config.long_prefill_token_threshold
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_num_seqs = config.max_num_seqs
        assert self.max_num_batched_tokens >= self.max_num_seqs # ensure that all reqs in decoding can be admitted.

        # slot
        self.req_slot_pool = RequestSlotPool(capacity=self.max_num_seqs)
        assert config.device is not None
        self.req_buf = RequestBuffer(cfg=config)
        self.tok_id_tab = TokenIdTable(max_num_seqs=self.max_num_seqs, max_model_len=config.max_model_len, device=config.device)
        self.sampling_param_tab = SamplingParamTable(self.config)

        # scheduling strategy
        self.use_d_first_schedule = config.use_d_first_schedule

        self.waiting: deque[ModelRequest] = deque()
        self.running: list[ModelRequest] = []
        self.cache = KVCache(config)

        # backoff after preempted
        self.backoff_base = config.backoff_base
        self.backoff_cap = config.backoff_cap

        # metrics
        self.sch_metrics = SchedulerMetrics()
        self.req_metrics_list: list[RequestMetrics] = []   # store temporarily
        self.req_metrics_interval = config.req_metrics_interval
        self._last_req_metrics_time = time.perf_counter()  # starting time

        # for benchmark
        self.is_benchmarking = config.is_benchmarking
        self.total_metrics: list[RequestMetrics] = []
        self.log_name_flag = datetime.now().strftime("%Y%m%d_%H%M%S")   # may be changed in test_benchmark_* functions.

    def teardown(self):
        self.cache.teardown()

    def add_request(self, req: ModelRequest) -> bool:
        max_want_blocks = cdiv(req.input_len + req.max_new_tokens, self.cache.block_size)
        if max_want_blocks > self.cache.pool.num_blocks:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"add_request error: request is too big, max_want_blocks: {max_want_blocks}, num_blocks: {self.cache.pool.num_blocks}")
            return False        # can never be served; reject at admission, not at allocation
        
        if len(self.waiting) >= self.max_waiting:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"add_request error, waiting exceeds, cur: {len(self.waiting)}, max_waiting: {self.max_waiting}")
            return False
        
        self.waiting.append(req)
        return True

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def _preempt(self, victim: ModelRequest):
        '''
        todo: swap out to host memory
        new issue: how to choose between swap out and recompute
        '''
        assert not victim.finished and not victim.projected_finished
        assert victim.slot is not None

        self.sch_metrics.report_on_preemption()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{victim.request_id} preempted, moved from running to waiting")

        self.running.remove(victim) if victim in self.running else None   # compatible for D_first_preemptive_schedule

        # must execute before reset_on_preemption because the original num_computed_tokens is needed for freeing cache
        self._free_resources(victim)

        victim.reset_projected_states_on_preemption()  # recompute from scratch, preempt_count += 1

        delay = min(self.backoff_base ** (victim.preempt_count - 1), self.backoff_cap)  # mininum of delay is 1
        victim.not_before_step = self.sch_metrics.step_id + delay
        self.waiting.appendleft(victim)

    def commit_step(self, sch_out: SchedulerOutput, num_truncated:int):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"call commit_step, ")
        fin = 0
        for req in sch_out.reqs:
            if req.finished:
                if not req.committed:
                    self.cleanup_on_finished(req=req)
                    fin += 1
                    req.committed = True
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"cleanup_on_finished, committed flips to true, {req.request_id}")
                else:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"skip cleanup_on_finished, {req.request_id}") # due to error/abort, already cleaned up

        self.sch_metrics.report_on_truncated(num_truncated)

        if sch_out.step_metrics is not None:
            sch_out.step_metrics.fin = fin      # number of finished reqs in this step

        sch_out.desc_num_in_flight()

    def cleanup_on_finished(self, req: ModelRequest):
        assert req.finished and req.projected_is_decoding
        assert req.slot is not None
        assert req in self.running

        self.sch_metrics.report_on_finish()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{req.request_id} finished, removed")

        self.running.remove(req)
        self._free_resources(req)
        self.req_metrics_list.append(req.metrics)

    def _do_cleanup_on_error(self, req: ModelRequest, e: Exception, is_from_running: bool = True):
        req.finished = True
        req.committed = True
        if req.loop is not None:
            req.loop.call_soon_threadsafe(req.token_queue.put_nowait, e)

        self._free_resources(req, is_from_running=is_from_running)
        self.req_metrics_list.append(req.metrics)
        self.sch_metrics.report_on_error()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"an error occured in {req.request_id}, removed")

    def cleanup_all_on_error(self, e: Exception):
        while self.running:
            self._do_cleanup_on_error(self.running.pop(), e)

        while self.waiting:
            self._do_cleanup_on_error(self.waiting.popleft(), e, is_from_running=False)

    def cleanup_on_error(self, req: ModelRequest, e: Exception):
        if req.committed:
            # partial items in sch_out/pre_sch_out succeeded, 
            # already cleaned up due to eos/max_new_tokens, so do nothing
            return

        if req in self.running:
            # from sch_out/pre_sch_out
            self.running.remove(req)
            self._do_cleanup_on_error(req, e)
        elif req in self.waiting:
            # from pre_sch_out, may be preempted and moved to waiting
            self.waiting.remove(req)
            self._do_cleanup_on_error(req, e, is_from_running=False)

    def cleanup_on_abort(self, request_id: str):
        '''called from api.py maybe due to the connection lost, but the request may have finished now'''
        req = None
        is_from_running = True
        for i in range(len(self.running)):
            if self.running[i].request_id == request_id:
                req = self.running.pop(i)
                break

        if req is None:
            for i in range(len(self.waiting)):
                if self.waiting[i].request_id == request_id:
                    # delete at index from deque
                    self.waiting.rotate(-i)
                    req = self.waiting.popleft()
                    self.waiting.rotate(i)
                    is_from_running = False
                    break

        if req is None:
            '''do not exist in both running and waiting, may have finished, so do nothing'''
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"the aborted request may have finished, do nothing, request_id: {request_id}")
            return

        self._do_cleanup_on_error(req,
            e=RuntimeError(f"request {request_id} aborted"),
            is_from_running=is_from_running)

    def _settle_and_check(self, r: ModelRequest) -> bool:
        # no in-flight batch will mutate r's actual states any more, so it can be re-admitted.
        if r.num_in_flight > 0:
            return False
        if r.is_preempted:
            # nothing will land to catch its actual states up with the projected ones: do it here.
            r.reset_actual_states_on_preemption()
            r.is_preempted = False
        return True

    def _pick_waiting(self) -> ModelRequest | None:
        '''
        todo:
        1) cdiv(len(input_ids) + expected_output_len, block_size) to avoid future preemption, 
              in which expected_output_len is evaluated by 50p of historical output lengths
        2) aging boost to avoid long prefills' starvation.
        '''
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"len, running: {len(self.running)}, waiting: {len(self.waiting)}")

        for r in self.waiting:
            if r.not_before_step <= self.sch_metrics.step_id and self._settle_and_check(r):
                return r

        # ignore backoff ONLY, to avoid starving the scheduler
        if not self.running and self.waiting:
            for r in self.waiting:
                if self._settle_and_check(r):
                    return r

        return None

    def _alloc_resources_on_admission(self, req: ModelRequest, want: int) -> list[int] | None:
        assert req.slot is None
        req.slot = self.req_slot_pool.alloc()
        self.tok_id_tab.add_req(req=req)    # holding prompt ids and output ids.
        self.sampling_param_tab.set_slot(req.slot, req.sampling, self.config)

        # allocate cache slots for admitted requests
        cache_slots = self.cache.allocate_slots(req, want=want, respect_watermark=True)
        if cache_slots is None:
            # free req slot when cache allocation fails,
            # otherwise the req slot will be leaked and the scheduler will receive an IndexError while the req pool exhausted.
            assert req.slot is not None
            self.req_slot_pool.free(req)

            self.sch_metrics.report_on_cache_exhausted()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"failed to allocate cache slots for {req.request_id}")
            return None

        return cache_slots

    def _free_resources(self, req: ModelRequest, is_from_running: bool = True):
        self.cache.free(req, use_assert=is_from_running)

        if is_from_running:
            self.req_slot_pool.free(req)

    def schedule(self, step_metrics: SchedulerStepMetrics | None = None) -> SchedulerOutput | None:            # called by run_loop
        self.sch_metrics.step_id += 1
        with timed(step_metrics, "sched"):
            if self.use_d_first_schedule:
                return self.D_first_preemptive_schedule(step_metrics)
            else:
                return self.preemptive_schedule(step_metrics)

    # D-first scheduling with no-cross preemption
    def D_first_preemptive_schedule(self, step_metrics: SchedulerStepMetrics | None = None) -> SchedulerOutput | None:            # called by run_loop when idle
        with timed(step_metrics, "sched_pre"):
            budget = self.max_num_batched_tokens
            scheduled: dict[str, ScheduledInfo] = {}

            # pre-process: ensure that all Ds are before all Ps
            decoding, prefill = [], []
            for req in list(self.running):
                (decoding if req.projected_is_decoding else prefill).append(req)
            self.running: list[ModelRequest] = decoding + prefill

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"len, running: {len(self.running)}, waiting: {len(self.waiting)}")

        # 1) running first — protect in-flight decodes' TPOT.
        with timed(step_metrics, "sched_run"):
            scheduled_running: list[ModelRequest] = []
            for req in list(self.running):
                # all finished requests should have been removed by cleanup_on_finished in commit_step after add_sampled_tokens
                assert not req.finished
                if req.projected_finished:
                    continue

                # the request that was preempted in the current or previous step should be skipped.
                if req.is_preempted:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"{req.request_id} has been prempted")
                    break
                
                want = 1 if req.projected_is_decoding else min(req.num_projected_prompt_remaining, budget, self.long_prefill_token_threshold)
                if want <= 0:
                    # only when in prefill (projected_is_decoding=False) and budget <= 0, which means all Ds has been scheduled and budget exhausted, the loop breaks.
                    # then kept the rest in running, but not be scheduled
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"no more budget for {req.request_id} to prefill, skip over")
                    break

                new_slots = self.cache.allocate_slots(req, want)
                while new_slots is None:       # KV pool exhausted
                    self.sch_metrics.report_on_cache_exhausted()

                    # traverse reversely, so it may have finished. it's the very gain over `preemptive_schedule`
                    victim = self.running.pop()
                    self._preempt(victim)     # yield no matter whether it's in decoding

                    if req.request_id == victim.request_id:     # cur req preempted
                        break

                    new_slots = self.cache.allocate_slots(req, want)

                if new_slots is None:               # still cannot get new blocks
                    break

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"add {req.request_id} to scheduled_running")
                scheduled_running.append(req)
                s_info = ScheduledInfo(want=want, cache_slots=new_slots)
                scheduled[req.request_id] = s_info
                budget -= want

        # 2) waiting next — fill remaining budget with (chunked) prefills
        with timed(step_metrics, "sched_wait"):
            while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
                req = self._pick_waiting()
                if req is None:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"not find a suitable waiting request, waiting len: {len(self.waiting)}")
                    break

                assert not req.projected_is_decoding
                want = min(req.num_projected_prompt_remaining, budget, self.long_prefill_token_threshold)
                cache_slots = self._alloc_resources_on_admission(req, want=want)    # alloc cache slots and req slot, if failed, return None
                if cache_slots is None:
                    break                                 # no room, stop admitting

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"move {req.request_id} from waiting to running and scheduled_running")
                scheduled_running.append(req)
                self.running.append(req)
                self.waiting.remove(req)
                s_info = ScheduledInfo(want=want, cache_slots=cache_slots)
                scheduled[req.request_id] = s_info
                budget -= want

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"schedule1, scheduled: {len(scheduled_running)}, running: {len(self.running)}, waiting: {len(self.waiting)}")

        if not scheduled_running:
            return None     # no work to do

        with timed(step_metrics, "sched_ret"):
            block_tables = []
            for req in scheduled_running:
                bt = self.cache.get_block_table(req)
                assert bt is not None
                block_tables.append(bt)

            self.sch_metrics.report_on_schedule(scheduled_reqs=scheduled_running)
            return SchedulerOutput(
                step_id=self.sch_metrics.step_id, reqs=scheduled_running,
                scheduled=scheduled, block_tables=block_tables,
                config=self.config, scheduler=self,
                step_metrics=step_metrics)
    
    def _pick_victim(self, cur_req):
        # pick strategies
        # 1) if cur_req is D, any P can be preempted. pick the newest D when there is no P.
        # 2) if cur_req is P, only Ps after cur_req can be preempted.
        for req in reversed(self.running):
            if not cur_req.projected_is_decoding and req.request_id == cur_req.request_id:  # strategy 2
                    return None

            if not req.projected_is_decoding:     # strategy 1
                return req

        # no P available: cur_req must be D, then apply strategy 1
        newest = self.running[-1]       # the last one, that is, the newest one.
        return None if newest.request_id == cur_req.request_id else newest

    # Preemptive scheduling with victim eviction
    def preemptive_schedule(self, step_metrics: SchedulerStepMetrics | None = None) -> SchedulerOutput | None:            # called by run_loop when idle
        with timed(step_metrics, "sched_pre"):
            budget = self.max_num_batched_tokens
            scheduled: dict[str, ScheduledInfo] = {}

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"len, running: {len(self.running)}, waiting: {len(self.waiting)}")
        
        # 1) running first — protect in-flight decodes' TPOT.
        #    Note: Ps and Ds may interleave.
        with timed(step_metrics, "sched_run"):
            scheduled_running: list[ModelRequest] = []
            for req in list(self.running):  # snapshot
                # all finished requests has been removed by cleanup_on_finished in commit_step after add_sampled_tokens
                assert not req.finished
                if req.projected_finished:
                    continue

                # the request that was preempted in the current or previous step should be skipped.
                if req.is_preempted:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"{req.request_id} has been prempted")
                    continue

                want = 1 if req.projected_is_decoding else min(req.num_projected_prompt_remaining, budget, self.long_prefill_token_threshold)
                if want <= 0:
                    # only when projected_is_decoding=False and budget <= 0, which means there is no room for current in-flight prefill request.
                    # then kept the rest in running, but not be scheduled
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"no more budget for {req.request_id} to prefill, skip over")
                    continue

                new_slots = self.cache.allocate_slots(req, want)
                while new_slots is None:                # KV pool exhausted
                    self.sch_metrics.report_on_cache_exhausted()

                    victim = self._pick_victim(req)
                    if victim is None:
                        self._preempt(req)     # yield no matter whether it's in decoding
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"{req.request_id} preempts itself")
                        break

                    self._preempt(victim)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"{req.request_id} preempts {victim.request_id}")

                    # roll back budget and blocks just allocated
                    s_info = scheduled.pop(victim.request_id, None)
                    if s_info is not None:  # only when victim is scheduled before.
                        scheduled_running.remove(victim)
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"remove {req.request_id} from scheduled_running")
                        budget += s_info.want

                    new_slots = self.cache.allocate_slots(req, want)

                if new_slots is None:                   # yield, according to "victim is None"
                    continue

                s_info = ScheduledInfo(want=want, cache_slots=new_slots)
                scheduled[req.request_id] = s_info
                scheduled_running.append(req)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"add to scheduled_running, step_id: {self.sch_metrics.step_id}, request_id: {req.request_id}, "
                                 f"num_scheduled_tokens: {req.num_scheduled_tokens}, num_computed_tokens: {req.num_computed_tokens}, want: {s_info.want}")
                budget -= want

        # 2) waiting next — fill remaining budget with (chunked) prefills
        with timed(step_metrics, "sched_wait"):
            while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
                req = self._pick_waiting()
                if req is None:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"not find a suitable waiting request, waiting len: {len(self.waiting)}")
                    break

                assert not req.projected_is_decoding
                want = min(req.num_projected_prompt_remaining, budget, self.long_prefill_token_threshold)
                cache_slots = self._alloc_resources_on_admission(req, want=want)    # alloc cache slots and req slot, if failed, return None
                if cache_slots is None:
                    break                                 # no room, stop admitting

                self.running.append(req)
                self.waiting.remove(req)
                s_info = ScheduledInfo(want=want, cache_slots=cache_slots)
                scheduled[req.request_id] = s_info
                scheduled_running.append(req)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"wating to scheduled_running, step_id: {self.sch_metrics.step_id}, request_id: {req.request_id}, "
                                 f"num_scheduled_tokens: {req.num_scheduled_tokens}, num_computed_tokens: {req.num_computed_tokens}, want: {s_info.want}")
                budget -= want

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"schedule1, scheduled: {len(scheduled_running)}, running: {len(self.running)}, waiting: {len(self.waiting)}")

        if not scheduled_running:
            return None     # no work to do

        with timed(step_metrics, "sched_ret"):
            block_tables = []
            for req in scheduled_running:
                bt = self.cache.get_block_table(req)
                assert bt is not None
                block_tables.append(bt)

            self.sch_metrics.report_on_schedule(scheduled_reqs=scheduled_running)
            return SchedulerOutput(
                step_id=self.sch_metrics.step_id, reqs=scheduled_running,
                scheduled=scheduled, block_tables=block_tables,
                config=self.config, scheduler=self,
                step_metrics=step_metrics)

    def log_metrics(self, sch_out: SchedulerOutput, is_exiting: bool=False):
        self.log_step_metrics(sch_out=sch_out)
        self.log_scheduler_metrics(is_exiting=is_exiting)

    def log_scheduler_metrics(self, is_exiting: bool=False):
        now = time.perf_counter()
        if now - self._last_req_metrics_time < self.req_metrics_interval and not is_exiting:
            return
        self._last_req_metrics_time = now

        if not self.req_metrics_list:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("empty req_metrics_list")
            return

        # snapshot
        tmp_counters: SchedulerMetrics = copy.deepcopy(self.sch_metrics)
        req_metrics_list = self.req_metrics_list
        self.req_metrics_list = []      # reset
        self.total_metrics.extend(req_metrics_list) if self.is_benchmarking else None

        if logger.isEnabledFor(logging.DEBUG):
            for metrics in req_metrics_list:
                json_bytes = orjson.dumps(round_floats(dataclasses.asdict(metrics), nd=3))
                logger.debug(f"metrics obj: {json_bytes.decode()}")

        # analyze
        json_bytes = analyze_metrics(req_metrics_list=req_metrics_list, sch_metrics=tmp_counters,
                                     is_benchmarking=self.is_benchmarking, config=self.config)
        logger.info(f"analyzed scheduler metrics: {json_bytes.decode()}")

        # save scheduler metrics to file
        log_path = Path(self.config.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        stats_path = log_path / f'sch_metrics.{self.log_name_flag}.json'
        with open(stats_path, "a+b") as f:
            f.write(json_bytes)
            f.write(b"\n")
            f.flush()

        if is_exiting and self.is_benchmarking:
            # save benchmark metrics to file
            assert len(self.total_metrics) > 0
            json_bytes = analyze_metrics(req_metrics_list=self.total_metrics,
                                        sch_metrics=self.sch_metrics,
                                        is_benchmarking=True, config=self.config)
            logger.info(f"benchmark_metrics: {json_bytes.decode()}")
            stats_log_file = log_path / f'benchmark_metrics.{self.log_name_flag}.json'
            with open(stats_log_file, "wb") as f:
                f.write(json_bytes)
                f.write(b"\n")
                f.flush()

    def log_step_metrics(self, sch_out: SchedulerOutput):
        if sch_out.step_metrics is None:
            return

        assert sch_out.step_metrics.is_stopped()

        # todo: Is it needed to roll back the committed requests due to eos in the previous step.

        json_bytes = orjson.dumps(round_floats(sch_out.step_metrics.output_dict(), nd=3))

        # save to file
        log_path = Path(self.config.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        stats_path = log_path / f'step_metrics.{self.log_name_flag}.json'
        with open(stats_path, "a+b") as f:
            f.write(json_bytes)
            f.write(b"\n")
            # f.flush()

