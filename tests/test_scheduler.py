import torch
import logging
import copy
import pytest
import time
import random
from dataclasses import dataclass

from qwen.config import ModelConfig
from qwen.scheduler import SchedulerOutput, ModelRequest, Scheduler, ScheduledInfo
from qwen.sampling import SamplingParams
from qwen.cache import cdiv
from qwen.metrics import SchedulerStepMetrics
from constants import TOK, TOK_EOS

logger = logging.getLogger(__name__)


@dataclass
class LifeCycle:
    sch: Scheduler
    req: ModelRequest
    s_info1: ScheduledInfo          # step 1: first prefill chunk
    s_info2: ScheduledInfo          # step 2: last prefill chunk, the batch still in flight
    pre_sch_out: SchedulerOutput    # that in-flight batch, committed one step later

@pytest.mark.parametrize("use_d_first_schedule", [True, False])
@pytest.mark.parametrize("is_finish_on_eos", [True, False])
def test_life_cycle(tmp_target_config: ModelConfig, is_finish_on_eos: bool, use_d_first_schedule: bool):
    '''mock a real workflow to verify the correctness of schedule1 and schedule2'''
    input_len = 100
    tmp_target_config.max_num_batched_tokens = 60
    tmp_target_config.max_num_seqs = 3
    tmp_target_config.num_blocks = 8
    tmp_target_config.block_size = 25

    tmp_target_config.max_waiting = 2
    tmp_target_config.cache_verification_interval = 0.  # always trigger cache invariant verification
    tmp_target_config.use_d_first_schedule = use_d_first_schedule

    sch = Scheduler(tmp_target_config)
    assert len(sch.req_slot_pool._free) == tmp_target_config.max_num_seqs == sch.req_slot_pool.capacity
    assert sch.sch_metrics.step_id == 0

    temperature = 1.0
    top_k = 3
    sampling = SamplingParams(temperature=temperature, top_k=top_k)
    assert sampling.temperature == temperature and sampling.top_k == top_k

    if is_finish_on_eos:
        max_new_tokens = 10 # do not finish at step 3
    else:
        max_new_tokens = 2  # finish at step 3

    # ---------- a new request ----------
    input_ids = [random.randrange(tmp_target_config.vocab_size) for _ in range(input_len)]
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids, sampling=sampling, request_id="req1", max_new_tokens=max_new_tokens)
    # verify default values
    assert req1.input_ids == input_ids and req1.input_len == len(input_ids)
    assert req1.sampling == sampling
    assert req1.max_new_tokens <= tmp_target_config.max_model_len-len(input_ids)
    assert req1.loop is None and req1.token_queue is not None
    assert req1.slot is None
    assert req1.is_preempted == False and req1.preempt_count == 0 and req1.not_before_step == 0
    assert req1.metrics is not None
    assert req1.output_ids == []
    assert req1.num_computed_tokens == 0 and req1.finished == False and req1.committed == False
    assert req1.num_scheduled_tokens == 0 and req1.projected_is_decoding == False and req1.projected_finished == False

    # ---------- add the new request to scheduler ----------
    assert sch.add_request(req=req1)
    assert sch.running == [] and list(sch.waiting) == [req1]
    assert req1.metrics.arrival_time is not None

    # ---------- step 1: schedule ----------
    sch_out = sch.schedule(step_metrics=SchedulerStepMetrics())
    assert sch_out is not None
    assert sch_out.reqs == [req1] and sch_out.batch_size == 1
    assert sch_out.needs_sample_device is None
    assert len(sch_out.s_infos) == 1
    s_info = sch_out.s_infos[0]
    assert s_info.want == tmp_target_config.max_num_batched_tokens
    assert cdiv(s_info.want, tmp_target_config.block_size) == 3
    assert len(s_info.cache_slots) == s_info.want
    assert sch_out.block_tables is not None and len(sch_out.block_tables)==1 and len(sch_out.block_tables[0])==3

    assert req1.metrics.first_schedule_time is not None
    assert req1.slot is not None
    assert req1.slot not in sch.req_slot_pool._free and len(sch.req_slot_pool._free) + 1 == tmp_target_config.max_num_seqs

    assert sch.tok_id_tab.num_computed_tok_device[req1.slot].item() == 0
    assert sch.tok_id_tab.in_len_device[req1.slot].item() == input_len
    assert sch.tok_id_tab.out_len_device[req1.slot].item() == 0
    assert sch.tok_id_tab.num_scheduled_tok_host[req1.slot].item() == 0
    assert sch.tok_id_tab.in_len_host[req1.slot].item() == input_len
    assert sch.tok_id_tab.projected_out_len_host[req1.slot].item() == 0

    assert sch_out.attn_meta is None
    assert sch_out.output_ids == [[]]
    assert sch_out.step_metrics is not None and sch_out.step_metrics.step_id == sch.sch_metrics.step_id == 1
    assert sch_out.dth_buf == 0

    # actual states
    assert req1.num_computed_tokens == 0
    assert req1.finished == False

    # ---------- step 1, mock forward / compute_logits / sampler  ----------
    # build attention metadata
    packed_ids, attn_meta = sch_out.build_attn_metadata(cache_data=sch.cache.data, rope=None)
    assert sch_out.slot_idx is not None and sch_out.slot_idx.tolist() == [req1.slot]
    assert sch_out.want is not None and sch_out.want.tolist() == [s_info.want]
    assert sch_out.needs_sample_device is not None and sch_out.needs_sample_device.tolist() == [False]
    assert packed_ids.tolist() == input_ids[:s_info.want]
    assert attn_meta.cos_sin is None    # due to None of rope
    assert attn_meta.cache == sch.cache.data
    assert attn_meta.max_seqlen_k == attn_meta.max_seqlen_q == s_info.want
    assert attn_meta.cu_seqlens_q.tolist() == [0, s_info.want] == attn_meta.cu_seqlens_k.tolist()
    assert attn_meta.cache_seqlens.tolist() == [s_info.want]

    assert attn_meta.position_ids.tolist() == list(range(0, s_info.want))
    assert attn_meta.block_table.tolist() == sch_out.block_tables
    assert attn_meta.slot_mapping.tolist() == s_info.cache_slots
    assert attn_meta.step_metrics_lst is not None and len(attn_meta.step_metrics_lst) == tmp_target_config.num_hidden_layers

    # build sampling tensors
    sampling_tensors = sch_out.build_sampling_tensors()
    assert sampling_tensors.top_k.tolist() == [top_k]
    torch.testing.assert_close(sampling_tensors.temperature, torch.tensor([temperature], device=tmp_target_config.device, dtype=torch.float32))
    torch.testing.assert_close(sampling_tensors.top_p, torch.tensor([tmp_target_config.top_p], device=tmp_target_config.device, dtype=torch.float32))

    # update host-side state
    sch_out.update_projected_state_in_advance()

    # projected states
    assert req1.num_scheduled_tokens == s_info.want
    assert req1.projected_is_decoding == False
    assert req1.projected_finished == False
    assert sch.tok_id_tab.num_scheduled_tok_host[req1.slot].item() == req1.num_scheduled_tokens
    assert sch.tok_id_tab.projected_out_len_host[req1.slot].item() == 0

    # add sampled tokens
    original_dth_buf = sch.tok_id_tab.dth_buf
    with pytest.raises(AssertionError):     # negative case
        next_tokens = torch.tensor([TOK, TOK], dtype=torch.int64, device=tmp_target_config.device)
        sch_out.dth_buf = sch.tok_id_tab.add_sampled_tokens_on_device(
            slot_idx=sch_out.slot_idx, needs_sample=sch_out.needs_sample_device,
            next_tokens=next_tokens, want=sch_out.want,
            step_metrics=sch_out.step_metrics)

    assert original_dth_buf == sch.tok_id_tab.dth_buf   # do not flip buffer after negative case

    # positive case
    next_tokens = torch.tensor([TOK], dtype=torch.int64, device=tmp_target_config.device)
    sch_out.dth_buf = sch.tok_id_tab.add_sampled_tokens_on_device(
            slot_idx=sch_out.slot_idx, needs_sample=sch_out.needs_sample_device,
            next_tokens=next_tokens, want=sch_out.want,
            step_metrics=sch_out.step_metrics)

    if torch.cuda.is_available():
        # do not check variables on device to avoid synchronization from device to host,
        # then num_computed_tok_device is not equal to num_scheduled_tok_host
        # assert sch.tok_id_tab.num_computed_tok_device[req1.slot].item() == 0
        # assert sch.tok_id_tab.out_len_device[req1.slot].item() == 0
        pass
    else:
        assert sch.tok_id_tab.num_computed_tok_device[req1.slot].item() == s_info.want
        assert sch.tok_id_tab.out_len_device[req1.slot].item() == 0

    # increase in-flight counter
    assert req1.num_in_flight == 0
    sch_out.incr_num_in_flight()
    assert req1.num_in_flight == 1
    
    # ---------- snapshot of step 1 ----------
    pre_sch_out = sch_out
    s_info1 = s_info
    pre_s_info = s_info

    # ---------- step 2: schedule ----------
    sch_out = sch.schedule(step_metrics=SchedulerStepMetrics())
    assert sch_out is not None
    assert sch_out.reqs == [req1] and sch_out.batch_size == 1
    assert sch_out.needs_sample_device is None
    assert len(sch_out.s_infos) == 1
    s_info = sch_out.s_infos[0]
    assert s_info.want == input_len - tmp_target_config.max_num_batched_tokens   # 40 = 100 - 60
    assert cdiv(input_len, tmp_target_config.block_size) == 4   # 100 / 25
    assert len(s_info.cache_slots) == s_info.want
    assert sch_out.block_tables is not None and len(sch_out.block_tables)==1 and len(sch_out.block_tables[0])==4

    assert req1.slot is not None
    assert req1.slot not in sch.req_slot_pool._free and len(sch.req_slot_pool._free) + 1 == tmp_target_config.max_num_seqs

    assert sch_out.attn_meta is None
    assert sch_out.output_ids == [[]]
    assert sch_out.step_metrics is not None and sch_out.step_metrics.step_id == sch.sch_metrics.step_id == 2    # step 2

    # actual states
    assert req1.num_computed_tokens == 0
    assert req1.finished == False

    # ---------- step 2, mock forward / compute_logits / sampler  ----------
    # build attention metadata
    packed_ids, attn_meta = sch_out.build_attn_metadata(cache_data=sch.cache.data, rope=None)
    assert sch_out.slot_idx is not None and sch_out.slot_idx.tolist() == [req1.slot]
    assert sch_out.want is not None and sch_out.want.tolist() == [s_info.want]
    assert sch_out.needs_sample_lst is not None and sch_out.needs_sample_lst == [True]
    assert sch_out.needs_sample_device is not None
    assert packed_ids.tolist() == input_ids[s_info1.want:]
    assert attn_meta.cos_sin is None    # due to None of rope
    assert attn_meta.cache == sch.cache.data
    assert attn_meta.max_seqlen_q == s_info.want
    assert attn_meta.cu_seqlens_q.tolist() == [0, s_info.want]
    assert attn_meta.max_seqlen_k == input_len
    assert attn_meta.cu_seqlens_k.tolist() == [0, input_len]
    assert attn_meta.cache_seqlens.tolist() == [input_len]

    assert attn_meta.position_ids.tolist() == list(range(s_info1.want, s_info1.want + s_info.want))
    assert attn_meta.block_table.tolist() == sch_out.block_tables
    assert attn_meta.slot_mapping.tolist() == s_info.cache_slots
    assert attn_meta.step_metrics_lst is not None and len(attn_meta.step_metrics_lst) == tmp_target_config.num_hidden_layers

    # build sampling tensors
    sampling_tensors = sch_out.build_sampling_tensors()
    assert sampling_tensors.top_k.tolist() == [top_k]
    torch.testing.assert_close(sampling_tensors.temperature, torch.tensor([temperature], device=tmp_target_config.device, dtype=torch.float32))
    torch.testing.assert_close(sampling_tensors.top_p, torch.tensor([tmp_target_config.top_p], device=tmp_target_config.device, dtype=torch.float32))

    # update host-side state
    sch_out.update_projected_state_in_advance()

    # projected states
    assert req1.num_scheduled_tokens == pre_s_info.want + s_info.want
    assert req1.projected_is_decoding == True       # flip because of the finish of prompt tokens
    assert req1.projected_finished == False
    assert sch.tok_id_tab.num_scheduled_tok_host[req1.slot].item() == req1.num_scheduled_tokens
    assert sch.tok_id_tab.projected_out_len_host[req1.slot].item() == 1

    # add sampled tokens
    next_tokens = torch.tensor([TOK], dtype=torch.int64, device=tmp_target_config.device)
    sch_out.dth_buf = sch.tok_id_tab.add_sampled_tokens_on_device(
            slot_idx=sch_out.slot_idx, needs_sample=sch_out.needs_sample_device,
            next_tokens=next_tokens, want=sch_out.want,
            step_metrics=sch_out.step_metrics)

    assert sch_out.dth_buf != pre_sch_out.dth_buf

    # ---------- mock that next_tokens of step 1 arrives at host from device ----------
    if torch.cuda.is_available():
        # verify in step 1
        assert pre_sch_out.step_metrics is not None
        pre_sch_out.step_metrics.events.synchronize("dth")
    else:
        # verify that in step 2 in advance on GPU
        assert sch.tok_id_tab.num_computed_tok_device[req1.slot].item() == req1.num_scheduled_tokens
        assert sch.tok_id_tab.out_len_device[req1.slot].item() == 1

    num_truncated = pre_sch_out.add_sampled_tokens_on_host()
    sch.commit_step(sch_out=pre_sch_out, num_truncated=num_truncated)

    # verify req1
    assert req1.output_ids == []
    # actual states
    assert req1.num_computed_tokens == s_info1.want
    assert req1.finished == False

    # increase in-flight counter
    assert req1.num_in_flight == 0
    sch_out.incr_num_in_flight()
    assert req1.num_in_flight == 1

    # ---------- snapshot of step 2 ----------
    ctx = LifeCycle(sch=sch, req=req1, s_info1=s_info1, s_info2=s_info, pre_sch_out=sch_out)

    if is_finish_on_eos:
        finish_on_eos(ctx)
    else:
        finish_on_max_new_token_limit(ctx)

def finish_on_eos(ctx: LifeCycle):
    sch, req1 = ctx.sch, ctx.req
    tmp_target_config = sch.config
    s_info1, s_info2 = ctx.s_info1, ctx.s_info2
    pre_sch_out, pre_s_info = ctx.pre_sch_out, s_info2      # the in-flight batch and its ScheduledInfo

    # ---------- step 3: schedule ----------
    sch_out = sch.schedule(step_metrics=SchedulerStepMetrics())
    assert sch_out is not None
    s_info = sch_out.s_infos[0]

    # ---------- step 3, mock forward / compute_logits / sampler  ----------
    # build attention metadata
    packed_ids, attn_meta = sch_out.build_attn_metadata(cache_data=sch.cache.data, rope=None)

    # build sampling tensors
    sampling_tensors = sch_out.build_sampling_tensors()

    # update host-side state
    sch_out.update_projected_state_in_advance()
    assert sch.tok_id_tab.num_scheduled_tok_host[req1.slot].item() == req1.num_scheduled_tokens
    assert sch.tok_id_tab.projected_out_len_host[req1.slot].item() == 2

    # add sampled tokens
    assert sch_out.slot_idx is not None and sch_out.needs_sample_device is not None
    assert sch_out.want is not None and sch_out.step_metrics is not None
    next_tokens = torch.tensor([TOK_EOS], dtype=torch.int64, device=tmp_target_config.device)
    sch_out.dth_buf = sch.tok_id_tab.add_sampled_tokens_on_device(
            slot_idx=sch_out.slot_idx, needs_sample=sch_out.needs_sample_device,
            next_tokens=next_tokens, want=sch_out.want,
            step_metrics=sch_out.step_metrics)

    # ---------- mock that next_tokens of step 2 arrives at host from device ----------
    if torch.cuda.is_available():
        # for the last step
        assert pre_sch_out.step_metrics is not None
        pre_sch_out.step_metrics.events.synchronize("dth")

    num_truncated = pre_sch_out.add_sampled_tokens_on_host()

    # verify req1
    assert req1.output_ids == [TOK]
    # actual states
    assert req1.num_computed_tokens == s_info1.want + pre_s_info.want
    assert req1.finished == False

    sch.commit_step(sch_out=pre_sch_out, num_truncated=num_truncated)

    # ---------- snapshot of step 3 ----------
    sch_out.incr_num_in_flight()
    pre_sch_out = sch_out
    s_info3 = s_info
    pre_s_info = s_info

    # ---------- step 4: schedule ----------
    sch_out = sch.schedule(step_metrics=SchedulerStepMetrics())
    assert sch_out is not None
    s_info = sch_out.s_infos[0]
    assert s_info.want == 1

    # ---------- step 4, mock forward / compute_logits / sampler  ----------
    # build attention metadata
    packed_ids, attn_meta = sch_out.build_attn_metadata(cache_data=sch.cache.data, rope=None)

    # build sampling tensors
    sampling_tensors = sch_out.build_sampling_tensors()

    # update host-side state
    sch_out.update_projected_state_in_advance()
    assert sch.tok_id_tab.num_scheduled_tok_host[req1.slot].item() == req1.num_scheduled_tokens
    assert sch.tok_id_tab.projected_out_len_host[req1.slot].item() == 3

    # add sampled tokens
    assert sch_out.slot_idx is not None and sch_out.needs_sample_device is not None
    assert sch_out.want is not None and sch_out.step_metrics is not None
    next_tokens = torch.tensor([TOK], dtype=torch.int64, device=tmp_target_config.device)
    sch_out.dth_buf = sch.tok_id_tab.add_sampled_tokens_on_device(
            slot_idx=sch_out.slot_idx, needs_sample=sch_out.needs_sample_device,
            next_tokens=next_tokens, want=sch_out.want,
            step_metrics=sch_out.step_metrics)

    # ---------- mock that next_tokens of step 3 arrives at host from device ----------
    if torch.cuda.is_available():
        # for the last step
        assert pre_sch_out.step_metrics is not None
        pre_sch_out.step_metrics.events.synchronize("dth")

    num_truncated = pre_sch_out.add_sampled_tokens_on_host()
    assert num_truncated == 0
    sch.commit_step(sch_out=pre_sch_out, num_truncated=num_truncated)

    # verify req1
    assert req1.output_ids == [TOK]
    # actual states
    assert req1.num_computed_tokens == s_info1.want + s_info2.want + pre_s_info.want
    assert req1.finished == True
    assert req1.committed == True
    assert req1 not in sch.running

    # ---------- snapshot of step 4 ----------
    sch_out.incr_num_in_flight()
    pre_sch_out = sch_out
    s_info4 = s_info
    pre_s_info = s_info

    # ---------- step 5: schedule ----------
    sch_out = sch.schedule(step_metrics=SchedulerStepMetrics())
    assert sch_out is None

    # ---------- mock that next_tokens of step 4 arrives at host from device ----------
    # req1 has been removed due to finish,
    # so it should be skipped in the following opeartions add_sampled_tokens_on_host and commit_step
    if torch.cuda.is_available():
        # for the last step
        assert pre_sch_out.step_metrics is not None
        pre_sch_out.step_metrics.events.synchronize("dth")

    num_truncated = pre_sch_out.add_sampled_tokens_on_host()
    sch.commit_step(sch_out=pre_sch_out, num_truncated=num_truncated)

    # verify req1
    assert req1.output_ids == [TOK]
    # actual states
    assert req1.num_computed_tokens == s_info1.want + s_info2.want + pre_s_info.want
    assert req1.finished == True
    assert req1 not in sch.running

    # all resource released
    assert req1.slot is None
    assert len(sch.req_slot_pool._free) == tmp_target_config.max_num_seqs
    assert len(sch.cache.pool.free) == tmp_target_config.num_blocks
    assert sch.sch_metrics.num_finished == 1

def finish_on_max_new_token_limit(ctx: LifeCycle):
    sch, req1 = ctx.sch, ctx.req
    tmp_target_config = sch.config
    input_len = req1.input_len
    assert req1.sampling is not None
    top_k, temperature = req1.sampling.top_k, req1.sampling.temperature
    s_info1, s_info2 = ctx.s_info1, ctx.s_info2
    pre_sch_out, pre_s_info = ctx.pre_sch_out, s_info2      # the in-flight batch and its ScheduledInfo

    # ---------- step 3: schedule ----------
    # req1.projected_is_decoding == True
    sch_out = sch.schedule(step_metrics=SchedulerStepMetrics())
    assert sch_out is not None
    assert sch_out.reqs == [req1] and sch_out.batch_size == 1
    assert sch_out.needs_sample_device is None
    assert len(sch_out.s_infos) == 1
    s_info = sch_out.s_infos[0]
    assert s_info.want == 1     # decoding
    assert cdiv(input_len+1, tmp_target_config.block_size) == 5   # 101 / 25
    assert len(s_info.cache_slots) == s_info.want
    assert sch_out.block_tables is not None and len(sch_out.block_tables)==1 and len(sch_out.block_tables[0])==5

    assert req1.slot is not None
    assert req1.slot not in sch.req_slot_pool._free and len(sch.req_slot_pool._free) + 1 == tmp_target_config.max_num_seqs

    assert sch_out.attn_meta is None
    assert sch_out.output_ids == [[]]
    assert sch_out.step_metrics is not None and sch_out.step_metrics.step_id == sch.sch_metrics.step_id == 3    # step 3

    # actual states
    assert req1.num_computed_tokens == s_info1.want
    assert req1.finished == False

    # ---------- step 3, mock forward / compute_logits / sampler  ----------
    # build attention metadata
    packed_ids, attn_meta = sch_out.build_attn_metadata(cache_data=sch.cache.data, rope=None)
    assert sch_out.slot_idx is not None and sch_out.slot_idx.tolist() == [req1.slot]
    assert sch_out.want is not None and sch_out.want.tolist() == [s_info.want]
    assert sch_out.needs_sample_lst is not None and sch_out.needs_sample_lst == [True]
    assert sch_out.needs_sample_device is not None
    assert packed_ids.numel() == 1
    assert attn_meta.cos_sin is None    # due to None of rope
    assert attn_meta.cache == sch.cache.data
    assert attn_meta.max_seqlen_q == s_info.want
    assert attn_meta.cu_seqlens_q.tolist() == [0, s_info.want]
    cache_len = input_len + s_info.want
    assert attn_meta.max_seqlen_k == cache_len
    assert attn_meta.cu_seqlens_k.tolist() == [0, cache_len]
    assert attn_meta.cache_seqlens.tolist() == [cache_len]

    assert attn_meta.position_ids.tolist() == list(range(s_info1.want+pre_s_info.want, s_info1.want+pre_s_info.want+s_info.want))
    assert attn_meta.block_table.tolist() == sch_out.block_tables
    assert attn_meta.slot_mapping.tolist() == s_info.cache_slots
    assert attn_meta.step_metrics_lst is not None and len(attn_meta.step_metrics_lst) == tmp_target_config.num_hidden_layers

    # build sampling tensors
    sampling_tensors = sch_out.build_sampling_tensors()
    assert sampling_tensors.top_k.tolist() == [top_k]
    torch.testing.assert_close(sampling_tensors.temperature, torch.tensor([temperature], device=tmp_target_config.device, dtype=torch.float32))
    torch.testing.assert_close(sampling_tensors.top_p, torch.tensor([tmp_target_config.top_p], device=tmp_target_config.device, dtype=torch.float32))

    # update host-side state
    sch_out.update_projected_state_in_advance()

    # projected states have been updated
    assert req1.num_scheduled_tokens == s_info1.want + s_info2.want + s_info.want
    assert req1.projected_is_decoding == True   # flip because of the finish of prompt tokens
    assert req1.projected_finished == True      # due to max_new_token=2
    assert sch.tok_id_tab.num_scheduled_tok_host[req1.slot].item() == req1.num_scheduled_tokens
    assert sch.tok_id_tab.projected_out_len_host[req1.slot].item() == 2

    # add sampled tokens
    next_tokens = torch.tensor([TOK+1], dtype=torch.int64, device=tmp_target_config.device)
    sch_out.dth_buf = sch.tok_id_tab.add_sampled_tokens_on_device(
            slot_idx=sch_out.slot_idx, needs_sample=sch_out.needs_sample_device,
            next_tokens=next_tokens, want=sch_out.want,
            step_metrics=sch_out.step_metrics)

    assert sch_out.dth_buf != pre_sch_out.dth_buf

    # ---------- mock that next_tokens of step 2 arrives at host from device ----------
    if torch.cuda.is_available():
        # verify in the last step
        assert pre_sch_out.step_metrics is not None
        pre_sch_out.step_metrics.events.synchronize("dth")
    else:
        # verify that in step 3 in advance on GPU
        assert sch.tok_id_tab.num_computed_tok_device[req1.slot].item() == req1.num_scheduled_tokens
        assert sch.tok_id_tab.out_len_device[req1.slot].item() == 2

    num_truncated = pre_sch_out.add_sampled_tokens_on_host()
    sch.commit_step(sch_out=pre_sch_out, num_truncated=num_truncated)

    # verify req1
    assert req1.output_ids == [TOK]
    # actual states
    assert req1.num_computed_tokens == s_info1.want + pre_s_info.want
    assert req1.finished == False

    # ---------- snapshot of step 3 ----------
    sch_out.incr_num_in_flight()
    pre_sch_out = sch_out
    s_info3 = s_info
    pre_s_info = s_info

    # ---------- step 4: schedule ----------
    sch_out = sch.schedule(step_metrics=SchedulerStepMetrics())
    assert sch_out is None

    # ---------- mock that next_tokens of step 3 arrives at host from device ----------
    if torch.cuda.is_available():
        # for the last step
        assert pre_sch_out.step_metrics is not None
        pre_sch_out.step_metrics.events.synchronize("dth")

    num_truncated = pre_sch_out.add_sampled_tokens_on_host()
    assert num_truncated == 1
    sch.commit_step(sch_out=pre_sch_out, num_truncated=num_truncated)

    # verify req1
    assert req1.output_ids == [TOK, TOK+1]
    # actual states
    assert req1.num_computed_tokens == s_info1.want + s_info2.want + pre_s_info.want
    assert req1.finished == True
    assert req1.projected_finished == True
    assert req1.committed == True
    assert req1 not in sch.running

    # all resource released
    assert req1.slot is None
    assert len(sch.req_slot_pool._free) == tmp_target_config.max_num_seqs
    assert len(sch.cache.pool.free) == tmp_target_config.num_blocks
    assert sch.sch_metrics.num_finished == 1


def test_gather_tok_in_recompute(tmp_target_config: ModelConfig):
    tmp_target_config.max_num_batched_tokens = 201
    tmp_target_config.num_blocks = 32

    sch = Scheduler(tmp_target_config)

    # req1: D, i_len=100, o_len=1, num_scheduled_tokens=100
    input_len = 100
    input_ids = [random.randrange(tmp_target_config.vocab_size) for _ in range(input_len)]
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids, request_id="req1")
    req1.num_computed_tokens = 0
    want = input_len
    sch._alloc_resources_on_admission(req=req1, want=want) # for slot and cache
    req1.num_computed_tokens = want
    req1.num_scheduled_tokens = want
    req1.output_ids = [TOK]
    req1.projected_is_decoding = True             # change to True because of num_computed_tokens==len(input_ids)

    # mock schedule
    req1.metrics.first_schedule_time = time.perf_counter()

    # preempted manually
    sch._free_resources(req=req1)
    req1.reset_projected_states_on_preemption()
    # req1: P, i_len=100, o_len=1, num_scheduled_tokens=0
    assert req1.projected_is_decoding == False    # default
    assert req1.num_scheduled_tokens == 0
    assert req1.is_preempted == True
    assert req1.metrics.first_schedule_time is not None     # do not reset first_schedule_time on preemption
    assert req1.output_ids == [TOK]

    if req1.is_preempted:
        req1.reset_actual_states_on_preemption()

    assert req1.num_computed_tokens == 0

    '''
    targeting bug: build_attn_metadata gets wrong input ids for preempt-then-recompute requests.
    '''
    all_toks = req1.input_ids + req1.output_ids
    mixed_prompt_len = len(req1.input_ids+req1.output_ids)

    req1.slot = sch.req_slot_pool.alloc()

    sch.tok_id_tab.add_req(req1)   # register to tokken id table.
    assert sch.tok_id_tab.in_len_device[req1.slot].item() == req1.input_len
    assert sch.tok_id_tab.out_len_device[req1.slot].item() == len(req1.output_ids)
    assert sch.tok_id_tab.num_computed_tok_device[req1.slot].item() == 0
    assert sch.tok_id_tab.in_len_host[req1.slot].item() == req1.input_len
    assert sch.tok_id_tab.projected_out_len_host[req1.slot].item() == len(req1.output_ids)
    assert sch.tok_id_tab.num_scheduled_tok_host[req1.slot].item() == 0
    assert sch.tok_id_tab.tok_id_device[req1.slot, :req1.input_len+len(req1.output_ids)].tolist() == all_toks

    slot_idx = torch.tensor([req1.slot], device=tmp_target_config.device, dtype=torch.int64)
    slot_idx_host = torch.tensor([req1.slot], dtype=torch.int64, device=torch.device("cpu")) \
        if torch.cuda.is_available() else slot_idx

    # case 1: [0:input_len+output_len]
    num_computed_tokens = 0
    sch.tok_id_tab.num_computed_tok_device[req1.slot] = num_computed_tokens
    sch.tok_id_tab.num_scheduled_tok_host[req1.slot] = num_computed_tokens
    want = mixed_prompt_len - num_computed_tokens
    want_tensor = torch.tensor([want], device=tmp_target_config.device, dtype=torch.int64)
    want_tensor_host = torch.tensor([want], device=torch.device("cpu"), dtype=torch.int64)
    packed_ids, position_ids, cu_seqlens_q, cache_seqlens, cu_seqlens_k, max_seqlen_k = \
        sch.tok_id_tab.gather_flat_pending_tok(slot_idx=slot_idx, slot_idx_stage=slot_idx_host,
                                               want=want_tensor, want_stage=want_tensor_host, num_tokens=want)
    assert packed_ids.tolist() == all_toks
    assert position_ids.tolist() == list(range(0, want))
    assert cu_seqlens_q.tolist() == [0, want]
    assert cache_seqlens.tolist() == [want]
    assert cu_seqlens_k.tolist() == [0, want]
    assert max_seqlen_k == want

    # case 2: [0:input_len]
    num_computed_tokens = 0
    sch.tok_id_tab.num_computed_tok_device[req1.slot] = num_computed_tokens
    sch.tok_id_tab.num_scheduled_tok_host[req1.slot] = num_computed_tokens
    want = req1.input_len
    want_tensor = torch.tensor([want], device=tmp_target_config.device, dtype=torch.int64)
    want_tensor_host = torch.tensor([want], device=torch.device("cpu"), dtype=torch.int64)
    packed_ids, position_ids, cu_seqlens_q, cache_seqlens, cu_seqlens_k, max_seqlen_k = \
        sch.tok_id_tab.gather_flat_pending_tok(slot_idx=slot_idx, slot_idx_stage=slot_idx_host,
                                               want=want_tensor, want_stage=want_tensor_host, num_tokens=want)
    assert packed_ids.tolist() == all_toks[num_computed_tokens:num_computed_tokens+want]
    assert position_ids.tolist() == list(range(num_computed_tokens, num_computed_tokens+want))

    # case 3: [1:input_len+output_len]
    num_computed_tokens = mixed_prompt_len - 1
    sch.tok_id_tab.num_computed_tok_device[req1.slot] = num_computed_tokens
    sch.tok_id_tab.num_scheduled_tok_host[req1.slot] = num_computed_tokens
    want = mixed_prompt_len - num_computed_tokens
    want_tensor = torch.tensor([want], device=tmp_target_config.device, dtype=torch.int64)
    want_tensor_host = torch.tensor([want], device=torch.device("cpu"), dtype=torch.int64)
    packed_ids, position_ids, cu_seqlens_q, cache_seqlens, cu_seqlens_k, max_seqlen_k = \
        sch.tok_id_tab.gather_flat_pending_tok(slot_idx=slot_idx, slot_idx_stage=slot_idx_host,
                                               want=want_tensor, want_stage=want_tensor_host, num_tokens=want)
    assert packed_ids.tolist() == all_toks[num_computed_tokens:num_computed_tokens+want]
    assert position_ids.tolist() == list(range(num_computed_tokens, num_computed_tokens+want))

    # case 4 (negative): [input_len+output_len:input_len+output_len], want=0
    num_computed_tokens = mixed_prompt_len
    sch.tok_id_tab.num_computed_tok_device[req1.slot] = num_computed_tokens
    sch.tok_id_tab.num_scheduled_tok_host[req1.slot] = num_computed_tokens
    want = mixed_prompt_len - num_computed_tokens
    want_tensor = torch.tensor([want], device=tmp_target_config.device, dtype=torch.int64)
    want_tensor_host = torch.tensor([want], device=torch.device("cpu"), dtype=torch.int64)
    with pytest.raises(AssertionError):     # negative case
        packed_ids, position_ids, cu_seqlens_q, cache_seqlens, cu_seqlens_k, max_seqlen_k = \
            sch.tok_id_tab.gather_flat_pending_tok(slot_idx=slot_idx, slot_idx_stage=slot_idx_host,
                                               want=want_tensor, want_stage=want_tensor_host, num_tokens=want)

    # case 5 (negative): [input_len:input_len+output_len+1], overflow
    num_computed_tokens = req1.input_len
    sch.tok_id_tab.num_computed_tok_device[req1.slot] = num_computed_tokens
    sch.tok_id_tab.num_scheduled_tok_host[req1.slot] = num_computed_tokens
    want = mixed_prompt_len - num_computed_tokens + 1
    want_tensor = torch.tensor([want], device=tmp_target_config.device, dtype=torch.int64)
    want_tensor_host = torch.tensor([want], device=torch.device("cpu"), dtype=torch.int64)
    with pytest.raises(ValueError):     # negative case
        packed_ids, position_ids, cu_seqlens_q, cache_seqlens, cu_seqlens_k, max_seqlen_k = \
            sch.tok_id_tab.gather_flat_pending_tok(slot_idx=slot_idx, slot_idx_stage=slot_idx_host,
                                               want=want_tensor, want_stage=want_tensor_host, num_tokens=want)


def _test_preemption_and_reschedule(sch: Scheduler, sch_out: SchedulerOutput, survival_req: ModelRequest, preempted_req: ModelRequest, tmp_target_config: ModelConfig):
    '''
    1. check running/waiting queue, metrics, and cache after preempted
    2. sruvial req generated EOS token, then finished, check its states, running/waiting queue, metrics, and cache
    3. submit a new request, put it to the tail of waiting; decrease num_max_seqs to 1 ensuring only one request in waiting can be scheduled.
    4. the leftest request in waiting just preempted is re-scheduled.
    Coverage: preempt, _pick_waiting, backoff
    '''
    # only survival_req succeeds
    assert sch.running == [survival_req]
    assert sch_out.reqs == [survival_req]
    assert len(sch_out.s_infos) == len(sch_out.reqs)
    assert survival_req.slot is not None

    # # preempted_req fails, and is preempted, which means that its kv block table is freed, 
    # # it is moved to waiting queue, and its intermediate states about forward are reset.
    assert preempted_req in list(sch.waiting)
    assert not preempted_req.projected_is_decoding
    assert preempted_req.slot is None
    assert preempted_req.num_scheduled_tokens == 0
    assert preempted_req.metrics.first_schedule_time is not None     # do not reset first_schedule_time on preemption
    assert preempted_req.preempt_count == 1 and preempted_req.not_before_step == sch.sch_metrics.step_id + 1     # not be delayed the first time it's preempted
    assert sch.cache.get_block_table(preempted_req) is None
    assert sch.sch_metrics.num_preempted == 1
    assert sch.sch_metrics.num_scheduled == 1
    assert sch.sch_metrics.num_cache_exhausted == 1

    # survival_req
    # standard process after schedule
    packed_ids, attn_meta = sch_out.build_attn_metadata(cache_data=sch.cache.data, rope=None)
    sampling_tensors = sch_out.build_sampling_tensors()
    sch_out.update_projected_state_in_advance()

    # add sampled tokens
    assert sch_out.slot_idx is not None
    assert sch_out.needs_sample_device is not None
    assert sch_out.want is not None and sch_out.step_metrics is not None
    next_tokens = torch.tensor([TOK_EOS], dtype=torch.int64, device=tmp_target_config.device)
    sch_out.dth_buf = sch.tok_id_tab.add_sampled_tokens_on_device(
            slot_idx=sch_out.slot_idx, needs_sample=sch_out.needs_sample_device,
            next_tokens=next_tokens, want=sch_out.want,
            step_metrics=sch_out.step_metrics)

    # mock queueing the in-flight batch
    sch_out.incr_num_in_flight()

    # mock dequeing the in-flight batch
    num_truncated = sch_out.add_sampled_tokens_on_host()
    assert survival_req.projected_is_decoding and survival_req.finished
    assert survival_req.slot is not None
    assert len(sch.req_slot_pool._free) + 1 == tmp_target_config.max_num_seqs
    assert survival_req in sch.running
    sch.commit_step(sch_out=sch_out, num_truncated=num_truncated)
    assert survival_req not in sch.running and survival_req not in list(sch.waiting) and sch.running == []
    assert sch.sch_metrics.num_finished == 1
    assert sch.cache.get_block_table(request=survival_req) is None      # kv cache released
    assert len(sch.cache.pool.free) == sch.cache.pool.num_blocks    # empty pool
    assert survival_req.slot is None    # slot released
    assert len(sch.req_slot_pool._free) == tmp_target_config.max_num_seqs

    # req3: P, i_len=30, o_len=0, num_scheduled_tokens=0
    input_ids = [random.randrange(tmp_target_config.vocab_size) for _ in range(30)]
    req3 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req3.request_id = "req3"
    req3.num_scheduled_tokens = 0        # want = tmp_target_config.max_num_batched_tokens - req1.want - req2.want
    assert req3.projected_is_decoding == False

    # submit req3 to waiting
    sch.waiting.append(req3)
    assert list(sch.waiting) == [preempted_req, req3]

    # WARNING: decrease max_num_seqs to 1, then only one rquest can be scheduled.
    sch.max_num_seqs = 1

    # req2, preempted before, will be scheduled because it sits on the leftest front of waiting deque.
    # test target: _pick_waiting
    sch_out2 = sch.schedule(step_metrics=SchedulerStepMetrics())
    assert sch_out2 is not None
    assert sch_out2.batch_size == 1 and sch_out2.reqs == [preempted_req]
    assert list(sch.waiting) == [req3]
    assert sch.sch_metrics.num_rescheduled == 1


@pytest.mark.parametrize("use_d_first_schedule", [True, False])
def test_preemption_in_prefill(tmp_target_config: ModelConfig, use_d_first_schedule: bool):
    '''
    1. two requests are both in prefill phase, and the newer one would be preempted while KV pool is exhausted.
    2. decrease num_max_seqs to 1 ensuring only one request in waiting can be scheduled.
    3. the leftest request in waiting just preempted is re-scheduled.
    '''
    tmp_target_config.max_num_batched_tokens = 201
    tmp_target_config.num_blocks = 3
    tmp_target_config.block_size = 16
    tmp_target_config.use_d_first_schedule = use_d_first_schedule

    sch = Scheduler(tmp_target_config)
    assert sch.cache.pool.num_blocks == tmp_target_config.num_blocks
    assert len(sch.cache.pool.free) == tmp_target_config.num_blocks

    # req1: P, i_len=23, o_len=0, num_scheduled_tokens=16
    input_ids = [random.randrange(tmp_target_config.vocab_size) for _ in range(23)]
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids, request_id="req1")
    req1.num_scheduled_tokens = 0
    want = 16
    sch._alloc_resources_on_admission(req1, want=want)      # construct its block table, and consume one block
    req1.num_scheduled_tokens = want
    req1.metrics.first_schedule_time = 1.
    assert req1.projected_is_decoding == False
    assert req1.slot is not None
    assert len(sch.req_slot_pool._free) + 1 == tmp_target_config.max_num_seqs

    # req2: P, i_len=20, o_len=0, num_scheduled_tokens=16
    input_ids = [random.randrange(tmp_target_config.vocab_size) for _ in range(20)]
    req2 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids, request_id="req2")
    req2.num_scheduled_tokens = 0
    want = 16
    sch._alloc_resources_on_admission(req2, want=want)      # construct its block table, and consume one block
    req2.num_scheduled_tokens = want
    req2.metrics.first_schedule_time = 1.
    assert req2.projected_is_decoding == False
    assert req2.slot is not None
    assert len(sch.req_slot_pool._free) + 2 == tmp_target_config.max_num_seqs

    sch.running = [req1, req2]

    # now there is only one free block left in kv cache pool
    assert len(sch.cache.pool.free) == 1

    # req1 requires a new block in this turn and is met, but there is no free block;
    # so req2's request is rejected due to the insufficiency of free blocks.
    sch_out = sch.schedule(step_metrics=SchedulerStepMetrics())
    assert sch_out is not None
    assert sch_out.batch_size == 1

    # only req1 succeeds
    assert sch_out.s_infos[0].want == 7
    table1 = sch.cache.get_block_table(req1)
    assert table1 is not None and len(table1) == 2
    assert len(sch.cache.pool.free) == 1

    _test_preemption_and_reschedule(sch=sch, sch_out=sch_out, survival_req=req1, preempted_req=req2, tmp_target_config=tmp_target_config)


@pytest.mark.parametrize("use_d_first_schedule", [True, False])
def test_preemption_in_decoding(tmp_target_config: ModelConfig, use_d_first_schedule: bool):
    '''
    1. two requests are in decoding phase, and the newer one would be preempted while KV pool is exhausted.
    2. decrease num_max_seqs to 1 ensuring only one request in waiting can be scheduled.
    3. the leftest request in waiting just preempted is re-scheduled.
    '''
    tmp_target_config.num_blocks = 3
    tmp_target_config.block_size = 16
    tmp_target_config.use_d_first_schedule = use_d_first_schedule

    sch = Scheduler(tmp_target_config)
    assert sch.cache.pool.num_blocks == tmp_target_config.num_blocks
    assert len(sch.cache.pool.free) == tmp_target_config.num_blocks

    # req1: D, i_len=23, o_len=1, num_scheduled_tokens=23
    input_ids = [random.randrange(tmp_target_config.vocab_size) for _ in range(23)]
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req1.request_id = "req1"
    req1.num_scheduled_tokens = 0
    sch._alloc_resources_on_admission(req1, 23)      # construct its block table, 2 blocks
    req1.num_scheduled_tokens = 23
    req1.output_ids = [TOK]
    req1.projected_is_decoding = True
    req1.metrics.first_schedule_time = 1.

    # req2: D, i_len=16, o_len=1, num_scheduled_tokens=16
    input_ids = [random.randrange(tmp_target_config.vocab_size) for _ in range(16)]
    req2 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req2.request_id = "req2"
    req2.num_scheduled_tokens = 0
    sch._alloc_resources_on_admission(req2, 16)     # construct its block table, 1 block
    req2.num_scheduled_tokens = 16
    req2.output_ids = [TOK]
    req2.projected_is_decoding = True
    req2.metrics.first_schedule_time = 1.

    sch.running = [req1, req2]

    # now there is no free blocks in kv cache pool
    assert len(sch.cache.pool.free) == 0

    # req1's reservation of kv blocks does not exhaust, so it can advance;
    # but req2's request for a new block is rejected due to the insufficiency of free blocks.
    sch_out = sch.schedule(step_metrics=SchedulerStepMetrics())
    assert sch_out is not None

    # only req1 succeeds
    assert sch_out.s_infos[0].want == 1
    table1 = sch.cache.get_block_table(req1)
    assert table1 is not None and len(table1) == 2
    assert len(sch.cache.pool.free) == 1    # freed by req2 due to preemption.

    _test_preemption_and_reschedule(sch=sch, sch_out=sch_out, survival_req=req1, preempted_req=req2, tmp_target_config=tmp_target_config)

@pytest.mark.parametrize("use_d_first_schedule", [True, False])
@pytest.mark.parametrize("p_before_d", [True, False])
def test_preemption_PD(tmp_target_config: ModelConfig, use_d_first_schedule: bool, p_before_d: bool):
    '''
    scenario: one P and one D in running queue in different orders, determined by p_before_d.
    expect: P is preempted by D due to the insufficiency of kv pool.
    '''
    tmp_target_config.num_blocks = 2
    tmp_target_config.block_size = 16
    tmp_target_config.use_d_first_schedule = use_d_first_schedule

    sch = Scheduler(tmp_target_config)
    assert sch.cache.pool.num_blocks == tmp_target_config.num_blocks
    assert len(sch.cache.pool.free) == tmp_target_config.num_blocks

    # req1: D, i_len=16, o_len=1, num_scheduled_tokens=16
    input_ids = [random.randrange(tmp_target_config.vocab_size) for _ in range(16)]
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req1.request_id = "req2"
    req1.num_scheduled_tokens = 0
    sch._alloc_resources_on_admission(req1, 16)     # construct its block table, 1 block
    req1.num_scheduled_tokens = 16
    req1.output_ids = [TOK]
    req1.projected_is_decoding = True
    req1.metrics.first_schedule_time = 1.

    # req2: P, i_len=23, o_len=0, num_scheduled_tokens=16
    input_ids = [random.randrange(tmp_target_config.vocab_size) for _ in range(23)]
    req2 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req2.request_id = "req1"
    req2.num_scheduled_tokens = 0
    sch._alloc_resources_on_admission(req2, 16)     # construct its block table, 1 block
    req2.num_scheduled_tokens = 16
    assert req2.projected_is_decoding == False
    req2.metrics.first_schedule_time = 1.

    sch.running = [req2, req1] if p_before_d else [req1, req2]

    # now there is no free blocks in kv cache pool
    assert len(sch.cache.pool.free) == 0

    # there is no free blocks for req1's and req2's requests.
    sch_out = sch.schedule(step_metrics=SchedulerStepMetrics())
    assert sch_out is not None

    # only req1 succeeds
    assert sch_out.s_infos[0].want == 1
    table2 = sch.cache.get_block_table(req1)
    assert table2 is not None and len(table2) == 2
    assert len(sch.cache.pool.free) == 0

    _test_preemption_and_reschedule(sch=sch, sch_out=sch_out, survival_req=req1, preempted_req=req2, tmp_target_config=tmp_target_config)

@pytest.mark.parametrize("use_d_first_schedule", [True, False])
def test_backoff(tmp_target_config: ModelConfig, use_d_first_schedule: bool):
    '''
    scenario: multiple requests in waiting, including some new requests, a preempted one with backoff in the front.
    expect: the preempted request not admitted in the following step.
    '''
    tmp_target_config.max_num_batched_tokens = 201
    tmp_target_config.num_blocks = 3
    tmp_target_config.block_size = 16
    tmp_target_config.use_d_first_schedule = use_d_first_schedule

    sch = Scheduler(tmp_target_config)
    assert sch.cache.pool.num_blocks == tmp_target_config.num_blocks
    assert len(sch.cache.pool.free) == tmp_target_config.num_blocks

    # req1: P, i_len=23, o_len=0, num_scheduled_tokens=16
    input_ids = [random.randrange(tmp_target_config.vocab_size) for _ in range(23)]
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req1.request_id = "req1"
    req1.num_scheduled_tokens = 0
    want = 16
    sch._alloc_resources_on_admission(req1, want=want)     # construct its block table, 1 block
    req1.num_scheduled_tokens = want
    req1.metrics.first_schedule_time = 1.
    assert req1.projected_is_decoding == False

    # frist time to preempt manually
    sch.running = [req1]
    sch._preempt(req1)
    assert sch.sch_metrics.num_preempted == 1
    assert req1.is_preempted
    assert req1.not_before_step == sch.sch_metrics.step_id+1
    assert req1 not in sch.running and req1 in list(sch.waiting)
    assert req1.num_scheduled_tokens == 0

    # second time to preempt manually. restore req1 first
    assert sch.waiting.popleft() == req1
    sch._alloc_resources_on_admission(req1, 16)
    req1.num_scheduled_tokens = 16
    sch.running = [req1]
    sch._preempt(req1)
    assert sch.sch_metrics.num_preempted == 2
    assert req1.is_preempted
    assert req1.not_before_step == sch.sch_metrics.step_id+2
    assert req1 not in sch.running and req1 in list(sch.waiting)

    # req3: P, i_len=30, o_len=0, num_scheduled_tokens=0
    input_ids = [random.randrange(tmp_target_config.vocab_size) for _ in range(30)]
    req3 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req3.request_id = "req3"
    req3.num_scheduled_tokens = 0        # want = tmp_target_config.max_num_batched_tokens - req1.want - req2.want
    assert req3.projected_is_decoding == False

    # submit req3 to waiting
    sch.waiting.append(req3)
    assert list(sch.waiting) == [req1, req3]

    # req1 not scheduled due to backoff
    sch_out = sch.schedule(step_metrics=SchedulerStepMetrics())
    assert sch_out is not None
    assert sch_out.reqs == [req3]
    assert req1 in list(sch.waiting)


@pytest.mark.parametrize("use_d_first_schedule", [True, False])
def test_aging_boost(tmp_target_config: ModelConfig, use_d_first_schedule: bool):
    '''
    todo
    scenario: multiple requests in running, and a request in waiting. sleep for a few seconds unitl the waiting time exceeds the aging threshold.
    expect: the waiting request is admitted.
    '''
    pass

def test_abort_and_error(tmp_target_config: ModelConfig):
    '''expect: error or aborted request removed from the running and waiting immediately'''
    tmp_target_config.num_blocks = 2
    tmp_target_config.block_size = 16

    sch = Scheduler(tmp_target_config)

    # req1: P, i_len=23, o_len=0, num_scheduled_tokens=16
    input_ids = [random.randrange(tmp_target_config.vocab_size) for _ in range(23)]
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req1.request_id = "req1"
    req1.num_scheduled_tokens = 0
    sch._alloc_resources_on_admission(req1, 16)     # construct its block table, 1 block
    req1.num_scheduled_tokens = 16
    assert req1.projected_is_decoding == False

    # req2: D, i_len=16, o_len=1, num_scheduled_tokens=16
    input_ids = [random.randrange(tmp_target_config.vocab_size) for _ in range(16)]
    req2 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req2.request_id = "req2"
    req2.num_scheduled_tokens = 0
    sch._alloc_resources_on_admission(req2, 16)     # construct its block table, 1 block
    req2.num_scheduled_tokens = 16
    req2.output_ids = [TOK]
    req2.projected_is_decoding = True

    sch.running = [req2, req1]

    sch.cleanup_on_abort(req1.request_id)
    assert req1 not in sch.running and req1 not in list(sch.waiting)
    assert req2 in sch.running
    assert sch.sch_metrics.num_error == 1

    sch.cleanup_on_error(req2, e=Exception())
    assert req2 not in sch.running and req2 not in list(sch.waiting)
    assert sch.sch_metrics.num_error == 2


