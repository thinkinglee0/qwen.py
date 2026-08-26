import torch
import logging
import copy
import pytest
import time

from qwen.config import ModelConfig
from qwen.scheduler import SchedulerOutput, ModelRequest, ScheduledInfo, Scheduler
from qwen.sampling import Sampling, TensorSampling
from qwen.cache import cdiv
from qwen.metrics import analyze_metrics, RequestMetrics, SchedulerMetrices

logger = logging.getLogger(__name__)

TOK = 100
TOK_EOS = 151643

def test_schedule(tmp_target_config: ModelConfig):
    '''mock a real workflow to verify the correctness of schedule1 and schedule2'''
    tmp_target_config.max_num_batched_tokens = 201
    tmp_target_config.max_num_seqs = 3
    tmp_target_config.num_blocks = 8
    tmp_target_config.max_waiting = 2
    tmp_target_config.cache_verification_interval = 0.  # always trigger cache invariant verification

    sch = Scheduler(tmp_target_config)

    temperature = 1.0
    top_k = 3
    sampling = Sampling(temperature=temperature, top_k=top_k)

    # Case 1: add requests in decoding, chunked prefill, and fully new, and reject one due to max_waiting
    # req1: D, i_len=100, o_len=1, num_computed_tokens=100
    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 100))[0].tolist()
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids, sampling=sampling)
    req1.request_id = "req1"
    req1.num_computed_tokens = 100      # want=1
    req1.output_ids = [99]
    assert req1.is_decoding == False    # default
    req1.is_decoding = True             # change to True because of num_computed_tokens==len(input_ids)

    # chunked prefill
    # req2: P, i_len=200, o_len=0, num_computed_tokens=100
    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 200))[0].tolist()
    req2 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids, sampling=sampling)
    req2.request_id = "req2"
    req2.num_computed_tokens = 100      # want=100
    assert req2.is_decoding == False

    # req3: P, i_len=300, o_len=0, num_computed_tokens=0
    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 300))[0].tolist()
    req3 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids, sampling=sampling)
    req3.request_id = "req3"
    req3.num_computed_tokens = 0        # want = tmp_target_config.max_num_batched_tokens - req1.want - req2.want
    assert req3.is_decoding == False

    # req4: P, i_len=200, o_len=0, num_computed_tokens=0
    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 200))[0].tolist()
    req4 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids, sampling=sampling)
    req4.request_id = "req4"
    req4.num_computed_tokens = 0
    assert req4.is_decoding == False

    # req5
    req5 = copy.deepcopy(req4)
    req5.request_id = "req5"

    # sch
    sch.running.append(req1)
    sch.running.append(req2)
    assert sch.add_request(req3)
    assert sch.add_request(req4)
    assert sch.add_request(req5) == False      # due to max_waiting=1
    assert sch.has_unfinished()
    assert sch.running == [req1, req2]
    assert list(sch.waiting) == [req3, req4]

    # step 1: verify D_first_preemptive_schedule, excepted want [1, 100, 100]
    # req1: D, i_len=100, o_len=1, num_computed_tokens=100
    # req2: P, i_len=200, o_len=0, num_computed_tokens=100
    # req3: P, i_len=300, o_len=0, num_computed_tokens=0, waiting -> running
    sch_out = sch.D_first_preemptive_schedule()
    assert sch.running == [req1, req2, req3]
    assert list(sch.waiting) == [req4]
    assert sch_out is not None
    assert sch_out.reqs == [req1, req2, req3]     # all scheduled, [D, P, P]
    assert sch_out.scheduled[req1.request_id].want == 1 and len(sch_out.scheduled[req1.request_id].slots) == 1  # decoding
    assert sch_out.scheduled[req2.request_id].want == len(req2.input_ids)-req2.num_computed_tokens and len(sch_out.scheduled[req2.request_id].slots) == len(req2.input_ids)-req2.num_computed_tokens
    assert sch_out.scheduled[req3.request_id].want == 100 and len(sch_out.scheduled[req3.request_id].slots) == 100  # 201-1-100

    # counters
    assert sch.sch_metrics.num_scheduled == 3
    assert sch.sch_metrics.num_rescheduled == 0
    assert sch.sch_metrics.num_preempted == 0
    assert sch.sch_metrics.num_finished == 0
    assert sch.sch_metrics.num_error == 0
    assert sch.sch_metrics.num_cache_exhausted == 0

    assert sch_out.output_ids == [[99], [], []]
    assert sch_out.finished == [False]*len(sch_out.reqs)

    # verify tensor_sampling
    assert sch_out.tensor_sampling.temperature is not None and sch_out.tensor_sampling.temperature.tolist() == [temperature]*3
    assert sch_out.tensor_sampling.top_k is not None and sch_out.tensor_sampling.top_k.tolist() == [top_k]*3

    # step 2: verify preemptive_schedule against D_first_preemptive_schedule
    # req1: D, i_len=100, o_len=1, num_computed_tokens=100
    # req2: P, i_len=200, o_len=0, num_computed_tokens=100
    # req3: P, i_len=300, o_len=0, num_computed_tokens=0
    for req in sch.running:
        req.metrics.first_schedule_time = None  # for counters
    sch.sch_metrics = SchedulerMetrices()   # reset counters
    sch_out2 = sch.preemptive_schedule()
    assert sch_out2 is not None
    assert sch_out.reqs == sch_out2.reqs
    assert sch_out.scheduled == sch_out2.scheduled
    torch.testing.assert_close(sch_out.tensor_sampling.temperature, sch_out2.tensor_sampling.temperature)
    torch.testing.assert_close(sch_out.tensor_sampling.top_k, sch_out2.tensor_sampling.top_k)

    # counters
    assert sch.sch_metrics.num_scheduled == 3
    assert sch.sch_metrics.num_rescheduled == 0
    assert sch.sch_metrics.num_preempted == 0
    assert sch.sch_metrics.num_finished == 0
    assert sch.sch_metrics.num_error == 0
    assert sch.sch_metrics.num_cache_exhausted == 0

    # step 3: add sampled tokens to the current batch after the mocked sampler, and verify intermediate states of all reqs
    # req1: D, i_len=100, o_len=1, num_computed_tokens=100
    # req2: P, i_len=200, o_len=0, num_computed_tokens=100
    # req3: P, i_len=300, o_len=0, num_computed_tokens=0
    # want [1, 100, 100]
    num_truncated = sch_out.add_sampled_tokens([100, 100, 100], tmp_target_config.eos_token_id_set)
    sch.commit_step(sch_out=sch_out, num_truncated=num_truncated)
    # req1: D, i_len=100, o_len=2, num_computed_tokens=101
    # req2: D, i_len=200, o_len=1, num_computed_tokens=200
    # req3: P, i_len=300, o_len=0, num_computed_tokens=100

    assert not req1.finished and not req2.finished and not req3.finished
    assert req1.is_decoding
    assert req2.is_decoding
    assert not req3.is_decoding
    assert req1.output_ids == [99, 100] and req2.output_ids == [100] and req3.output_ids == []
    assert req1.num_computed_tokens == len(req1.input_ids) + len(req1.output_ids) - 1
    assert req2.num_computed_tokens == len(req2.input_ids)
    assert req3.num_computed_tokens == 100

    # step 4: next scheduling, and expected want [1, 1, 199]
    # req1: D, i_len=100, o_len=2, num_computed_tokens=101
    # req2: D, i_len=200, o_len=1, num_computed_tokens=200
    # req3: P, i_len=300, o_len=0, num_computed_tokens=100
    sch_out = sch.D_first_preemptive_schedule()
    assert sch_out is not None
    assert sch_out.reqs == [req1, req2, req3]     # all scheduled, [D, D, P]
    assert sch_out.scheduled[req1.request_id].want == 1 and len(sch_out.scheduled[req1.request_id].slots) == 1  # decoding
    assert sch_out.scheduled[req2.request_id].want == 1 and len(sch_out.scheduled[req2.request_id].slots) == 1
    assert sch_out.scheduled[req3.request_id].want == 199 and len(sch_out.scheduled[req3.request_id].slots) == 199  # 201-1-1

    # verify pending tokens
    assert req1.get_existing_ids(sch_out.scheduled[req1.request_id].want) == [100]
    assert req2.get_existing_ids(sch_out.scheduled[req2.request_id].want) == [100]
    assert req3.get_existing_ids(sch_out.scheduled[req3.request_id].want) == req3.input_ids[req3.num_computed_tokens:req3.num_computed_tokens+199]

    # verify the batch's intermediate states
    assert sch_out.output_ids == [[99, 100], [100], []]
    assert sch_out.finished == [False]*len(sch_out.reqs)

    # step 5: req1 comes across EOS. add sampled tokens to the current batch, then update states of reqs
    # want [1, 1, 199]
    # req1: D, i_len=100, o_len=2, num_computed_tokens=101
    # req2: D, i_len=200, o_len=1, num_computed_tokens=200
    # req3: P, i_len=300, o_len=0, num_computed_tokens=100
    num_truncated = sch_out.add_sampled_tokens([TOK_EOS, 101, 101], tmp_target_config.eos_token_id_set)
    sch.commit_step(sch_out=sch_out, num_truncated=num_truncated)
    # req1: D, i_len=100, o_len=2, num_computed_tokens=102, finished, removed from running
    # req2: D, i_len=200, o_len=2, num_computed_tokens=201
    # req3: P, i_len=300, o_len=0, num_computed_tokens=299

    # verify intermediate states of reqs
    # req1 finished, is removed from running, and its block table was freed.
    assert req1.finished and not req2.finished and not req3.finished
    assert req1.is_decoding and req2.is_decoding and not req3.is_decoding

    logger.info([r.request_id for r in sch.running])
    assert req1 not in sch.running
    assert sch.sch_metrics.num_finished == 1
    assert len(sch.req_metrics_list) == 1
    assert sch.cache.get_block_table(req1) is None

    assert req1.output_ids == [99, 100] and req2.output_ids == [100, 101] and req3.output_ids == []
    assert req1.num_computed_tokens == 2+len(req1.input_ids)
    assert req2.num_computed_tokens == 1+len(req2.input_ids)
    assert req3.num_computed_tokens == 299

    # step 6: add a new request `req5`, and verify that the finished request `req1` have been removed
    # req2: D, i_len=200, o_len=2, num_computed_tokens=201
    # req3: P, i_len=300, o_len=0, num_computed_tokens=299
    # req4: P, i_len=200, o_len=0, num_computed_tokens=0, waiting
    assert sch.add_request(req5)      # input_ids len: 100
    assert list(sch.waiting) == [req4, req5]
    assert not req4.is_decoding and not req5.is_decoding
    sch_out = sch.D_first_preemptive_schedule()    # expected want [1, 1, 199]
    # req4: P, i_len=200, o_len=0, num_computed_tokens=0, waiting -> running

    assert sch.running == [req2, req3, req4]
    assert list(sch.waiting) == [req5]
    assert sch_out is not None
    assert sch_out.reqs == [req2, req3, req4]     # req1 removed.
    assert sch_out.scheduled[req2.request_id].want == 1 and len(sch_out.scheduled[req2.request_id].slots) == 1
    assert sch_out.scheduled[req3.request_id].want == 1 and len(sch_out.scheduled[req3.request_id].slots) == 1  # min(201-1, 300-299)
    assert sch_out.scheduled[req4.request_id].want == 199 and len(sch_out.scheduled[req4.request_id].slots) == 199  # min(201-2, 200)

    assert req2.get_existing_ids(sch_out.scheduled[req2.request_id].want) == [101]
    assert req3.get_existing_ids(sch_out.scheduled[req3.request_id].want) == req3.input_ids[-1:]   # last element
    assert req4.get_existing_ids(sch_out.scheduled[req4.request_id].want) == req4.input_ids[:199]  # top xx elements

    assert sch_out.output_ids == [[100, 101], [], []]
    assert sch_out.finished == [False]*len(sch_out.reqs)

    # step 7: verify the invariant of D-before-P order
    # change the order of reqs in the running list manually
    # req2: D, i_len=200, o_len=2, num_computed_tokens=201
    # req3: P, i_len=300, o_len=0, num_computed_tokens=299
    # req4: P, i_len=200, o_len=0, num_computed_tokens=0
    sch.running = [req4, req2, req3]
    sch_out = sch.D_first_preemptive_schedule()       # expected want [1, 200]
    assert sch_out is not None
    assert sch.running == [req2, req4, req3]    # [req2:D, req4:P, req3:P], keep invariant of D-before-P
    assert sch_out.reqs == [req2, req4]     # D-before-P, req3 not scheduled due to budget, but keptd in running
    assert sch_out.scheduled[req2.request_id].want == 1 and len(sch_out.scheduled[req2.request_id].slots) == 1
    assert sch_out.scheduled[req4.request_id].want == 200 and len(sch_out.scheduled[req4.request_id].slots) == 200  # min(201-1, 200)

    # step 8: verify preemptive_schedule
    # req2: D, i_len=200, o_len=2, num_computed_tokens=201
    # req3: P, i_len=300, o_len=0, num_computed_tokens=299
    # req4: P, i_len=200, o_len=0, num_computed_tokens=0
    sch.running = [req3, req4, req2]
    sch.req_metrics_list = []
    for req in sch.running:
        req.metrics.first_schedule_time = None  # for counters
    sch.sch_metrics = SchedulerMetrices()

    sch_out = sch.preemptive_schedule()    # expected want [1, 200, 1]
    assert sch_out is not None
    assert sch_out.reqs == [req3, req4, req2]     # req2 still scheduled although budget<0
    assert sch_out.scheduled[req3.request_id].want == 1 and len(sch_out.scheduled[req3.request_id].slots) == 1
    assert sch_out.scheduled[req4.request_id].want == 200 and len(sch_out.scheduled[req4.request_id].slots) == 200  # min(201-1, 200)
    assert sch_out.scheduled[req2.request_id].want == 1 and len(sch_out.scheduled[req2.request_id].slots) == 1
    assert sum([si.want for si in sch_out.scheduled.values()]) > tmp_target_config.max_num_batched_tokens
    
    # counters
    assert sch.sch_metrics.num_scheduled == 3
    assert sch.sch_metrics.num_rescheduled == 0
    assert sch.sch_metrics.num_preempted == 0
    assert sch.sch_metrics.num_finished == 0
    assert sch.sch_metrics.num_error == 0
    assert sch.sch_metrics.num_cache_exhausted == 0

@pytest.mark.parametrize("use_d_first_schedule", [True, False])
def test_recompute(tmp_target_config: ModelConfig, use_d_first_schedule: bool):
    tmp_target_config.max_num_batched_tokens = 201
    tmp_target_config.num_blocks = 32
    tmp_target_config.use_d_first_schedule=use_d_first_schedule

    sch = Scheduler(tmp_target_config)

    # req1: D, i_len=100, o_len=1, num_computed_tokens=100
    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 100))[0].tolist()
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req1.request_id = "req1"
    req1.num_computed_tokens = 0
    sch.cache.allocate_slots(req1, 100) # construct its block table
    req1.num_computed_tokens = 100      # will want=1 in next schedule
    req1.output_ids = [TOK]
    req1.is_decoding = True             # change to True because of num_computed_tokens==len(input_ids)

    # mock schedule
    req1.metrics.first_schedule_time = time.perf_counter()

    # preempted manually
    req1.reset_on_preemption()
    # req1: P, i_len=100, o_len=1, num_computed_tokens=0
    assert req1.is_decoding == False    # default
    assert req1.num_computed_tokens == 0
    assert req1.metrics.first_schedule_time is not None     # do not reset first_schedule_time on preemption

    # req2: P, i_len=200, o_len=0, num_computed_tokens=100
    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 200))[0].tolist()
    req2 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req2.request_id = "req2"
    req2.num_computed_tokens = 0
    sch.cache.allocate_slots(req2, 100) # construct its block table
    req2.num_computed_tokens = 100      # want=100
    assert req2.is_decoding == False

    sch.running = [req2]
    sch.waiting.append(req1)

    # req2: P, i_len=200, o_len=0, num_computed_tokens=100
    # req1: P, i_len=100, o_len=1, num_computed_tokens=0, waiting
    sch_out = sch.schedule()
    assert sch_out is not None
    assert sch_out.reqs == [req2, req1]
    assert sch_out.scheduled[req2.request_id].want == 100   # 200-100
    assert sch_out.scheduled[req1.request_id].want == 101   # i_len+o_len=100+1=101
    assert req1.metrics.first_schedule_time is not None and req2.metrics.first_schedule_time is not None
    assert req1.metrics.first_schedule_time < req2.metrics.first_schedule_time

    # counters
    assert sch.sch_metrics.num_scheduled == 2
    assert sch.sch_metrics.num_rescheduled == 1

    # after sampling
    num_truncated = sch_out.add_sampled_tokens([TOK, TOK], tmp_target_config.eos_token_id_set)
    sch.commit_step(sch_out=sch_out, num_truncated=num_truncated)
    # req2: P, i_len=200, o_len=1, num_computed_tokens=200
    # req1: P, i_len=100, o_len=2, num_computed_tokens=101
    assert req2.is_decoding and req1.is_decoding
    assert req2.num_computed_tokens == 200 and req1.num_computed_tokens == 101
    assert req2.output_ids == [TOK] and req1.output_ids == [TOK, TOK]


@pytest.mark.parametrize("use_d_first_schedule", [True, False])
def test_preemption_in_prefill(tmp_target_config: ModelConfig, use_d_first_schedule: bool):
    '''two requests are both in prefill phase, and the newer one would be preempted while KV pool is exhausted.'''
    tmp_target_config.max_num_batched_tokens = 201
    tmp_target_config.num_blocks = 3
    tmp_target_config.block_size = 16
    tmp_target_config.use_d_first_schedule = use_d_first_schedule

    sch = Scheduler(tmp_target_config)
    assert sch.cache.pool.num_blocks == tmp_target_config.num_blocks
    assert len(sch.cache.pool.free) == tmp_target_config.num_blocks

    # req1: P, i_len=23, o_len=0, num_computed_tokens=16
    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 23))[0].tolist()
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req1.request_id = "req1"
    req1.num_computed_tokens = 0
    sch.cache.allocate_slots(req1, 16)      # construct its block table, and consume one block
    req1.num_computed_tokens = 16
    assert req1.is_decoding == False

    # req2: P, i_len=100, o_len=0, num_computed_tokens=16
    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 100))[0].tolist()
    req2 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req2.request_id = "req2"
    req2.num_computed_tokens = 0
    sch.cache.allocate_slots(req2, 16)      # construct its block table, and consume one block
    req2.num_computed_tokens = 16
    assert req2.is_decoding == False

    sch.running = [req1, req2]

    # now there is only one free block in kv cache pool
    assert len(sch.cache.pool.free) == 1

    # req1 requires a new block in this turn and is met, but there is no free block;
    # so req2's request is rejected due to the insufficiency of free blocks.
    sch_out = sch.schedule()
    assert sch_out is not None

    # only req1 succeeds
    assert sch.running == [req1]
    assert sch_out.reqs == [req1]
    assert sch_out.scheduled[req1.request_id].want == 7
    table1 = sch.cache.get_block_table(req1)
    assert table1 is not None and len(table1) == 2
    assert len(sch.cache.pool.free) == 1

    # req2 fails, and is preempted
    assert sch_out.scheduled.get(req2.request_id) is None
    assert req2 in list(sch.waiting)
    assert not req2.is_decoding
    assert req2.num_computed_tokens == 0
    assert sch.cache.get_block_table(req2) is None
    assert sch.sch_metrics.num_preempted == 1
    assert sch.sch_metrics.num_scheduled == 1
    assert sch.sch_metrics.num_cache_exhausted == 1

@pytest.mark.parametrize("use_d_first_schedule", [True, False])
def test_preemption_in_decoding(tmp_target_config: ModelConfig, use_d_first_schedule: bool):
    '''two requests are in decoding phase, and the newer one would be preempted while KV pool is exhausted.'''
    tmp_target_config.num_blocks = 3
    tmp_target_config.block_size = 16
    tmp_target_config.use_d_first_schedule = use_d_first_schedule

    sch = Scheduler(tmp_target_config)
    assert sch.cache.pool.num_blocks == tmp_target_config.num_blocks
    assert len(sch.cache.pool.free) == tmp_target_config.num_blocks

    # req1: D, i_len=23, o_len=1, num_computed_tokens=23
    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 23))[0].tolist()
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req1.request_id = "req1"
    req1.num_computed_tokens = 0
    sch.cache.allocate_slots(req1, 23)      # construct its block table, 2 blocks
    req1.num_computed_tokens = 23
    req1.output_ids = [TOK]
    req1.is_decoding = True

    # req2: D, i_len=16, o_len=1, num_computed_tokens=16
    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 16))[0].tolist()
    req2 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req2.request_id = "req2"
    req2.num_computed_tokens = 0
    sch.cache.allocate_slots(req2, 16)      # construct its block table, 1 block
    req2.num_computed_tokens = 16
    req2.output_ids = [TOK]
    req2.is_decoding = True

    sch.running = [req1, req2]

    # now there is no free blocks in kv cache pool
    assert len(sch.cache.pool.free) == 0

    # req1's reservation of kv blocks does not exhaust, so it can advance;
    # but req2's request for a new block is rejected due to the insufficiency of free blocks.
    sch_out = sch.schedule()
    assert sch_out is not None

    # only req1 succeeds
    assert sch.running == [req1]
    assert sch_out.reqs == [req1]
    assert sch_out.scheduled[req1.request_id].want == 1
    table1 = sch.cache.get_block_table(req1)
    assert table1 is not None and len(table1) == 2
    assert len(sch.cache.pool.free) == 1    # freed by req2 due to preemption.

    # req2 fails, and is preempted, which means that its kv block table is freed, 
    # it is moved to waiting queue, and its intermediate states about forward are reset.
    assert sch_out.scheduled.get(req2.request_id) is None
    assert req2 in list(sch.waiting)
    assert not req2.is_decoding
    assert req2.num_computed_tokens == 0
    assert sch.cache.get_block_table(req2) is None

    # counters
    assert sch.sch_metrics.num_preempted == 1
    assert sch.sch_metrics.num_scheduled == 1
    assert sch.sch_metrics.num_cache_exhausted == 1

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

    # req1: P, i_len=23, o_len=0, num_computed_tokens=16
    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 23))[0].tolist()
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req1.request_id = "req1"
    req1.num_computed_tokens = 0
    sch.cache.allocate_slots(req1, 16)      # construct its block table, 1 block
    req1.num_computed_tokens = 16
    assert req1.is_decoding == False

    # req2: D, i_len=16, o_len=1, num_computed_tokens=16
    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 16))[0].tolist()
    req2 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids)
    req2.request_id = "req2"
    req2.num_computed_tokens = 0
    sch.cache.allocate_slots(req2, 16)      # construct its block table, 1 block
    req2.num_computed_tokens = 16
    req2.output_ids = [TOK]
    req2.is_decoding = True

    sch.running = [req1, req2] if p_before_d else [req2, req1]

    # now there is no free blocks in kv cache pool
    assert len(sch.cache.pool.free) == 0

    # there is no free blocks for req1's and req2's requests.
    sch_out = sch.schedule()
    assert sch_out is not None

    # only req2 succeeds
    assert sch.running == [req2]
    assert sch_out.reqs == [req2]
    assert sch_out.scheduled[req2.request_id].want == 1
    table2 = sch.cache.get_block_table(req2)
    assert table2 is not None and len(table2) == 2
    assert len(sch.cache.pool.free) == 0

    # req1 fails, and is preempted, which means that its kv block table is freed, 
    # it is moved to waiting queue, and its intermediate states about forward are reset.
    assert sch_out.scheduled.get(req1.request_id) is None
    assert req1 in list(sch.waiting)
    assert not req1.is_decoding
    assert req1.num_computed_tokens == 0
    assert sch.cache.get_block_table(req1) is None

    # counters
    assert sch.sch_metrics.num_scheduled == 1
    assert sch.sch_metrics.num_preempted == 1
    assert sch.sch_metrics.num_cache_exhausted == 1

@pytest.mark.parametrize("use_d_first_schedule", [True, False])
def test_pick_waiting(tmp_target_config: ModelConfig, use_d_first_schedule: bool):
    '''
    todo
    scenario: multiple requests in waiting, including some new requests, a preempted victim in the front.
    expect: the preempted request not admitted in the following step.
    targeting bug: waiting.popleft will remove the leftest request when _pick_waiting picks non-leftest requests.
    '''
    pass

@pytest.mark.parametrize("use_d_first_schedule", [True, False])
def test_backoff(tmp_target_config: ModelConfig, use_d_first_schedule: bool):
    '''
    todo
    scenario: multiple requests in waiting, including some new requests, a preempted one with backoff in the front.
    expect: the preempted request not admitted in the following step.
    '''
    pass

@pytest.mark.parametrize("use_d_first_schedule", [True, False])
def test_aging_boost(tmp_target_config: ModelConfig, use_d_first_schedule: bool):
    '''
    todo
    scenario: multiple requests in running, and a request in waiting. sleep for a few seconds unitl the waiting time exceeds the aging threshold.
    expect: the waiting request is admitted.
    '''
    pass



