from collections import deque
import logging
from dataclasses import dataclass, asdict
import asyncio
import time
from datetime import datetime
from pathlib import Path
import copy
import orjson
import uuid

from qwen.utils import round_floats
from qwen.sampling import TensorSampling, Sampling
from qwen.config import ModelConfig
from qwen.metrics import analyze_metrics, RequestMetrics, SchedulerMetrices, SchedulerStepMetrices
from qwen.cache import KVCache, cdiv

logger = logging.getLogger(__name__)


class ModelRequest:
    def __init__(self, config: ModelConfig, loop, input_ids: list[int], request_id: str | None=None, sampling: Sampling | None = None, max_new_tokens: int=1024):
        self.request_id = request_id if request_id is not None else str(uuid.uuid4())
        self.input_ids = input_ids
        self.sampling = sampling
        self.max_new_tokens = max(1, min(config.max_model_len-len(input_ids), max_new_tokens))
        self.loop = loop
        self.token_queue: asyncio.Queue[int | None | Exception] = asyncio.Queue()

        # for backoff
        self.preempt_count: int = 0
        self.not_before_step: int = 0     # earliest scheduler step at which re-admission is allowed

        # metrics
        self.metrics = RequestMetrics(arrival_time=time.perf_counter(), num_input_token=len(self.input_ids))

        # intermediate states
        self.num_computed_tokens: int = 0   # for kv cache
        self.is_decoding: bool = False      # P or D phase
        self.preempt_count = 0
        self.output_ids: list[int] = []
        self.finished: bool = False

    # keep output_ids for consistency from user's perspective.
    # only drop states about kv cache and queuing
    def reset_on_preemption(self) -> None:
        self.num_computed_tokens = 0    # for kv cache
        self.is_decoding = False        # phase
        self.preempt_count += 1

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
    def _num_prefill_tokens(self) -> int:
        if not self.is_decoding:
            return len(self.input_ids) + len(self.output_ids)
        else:
            return len(self.input_ids)

    @property
    def num_prompt_remaining(self) -> int:  # compatible with requests re-computing from scratch after evicted from decoding
        return max(0, self._num_prefill_tokens - self.num_computed_tokens)

    # return 1 if this request is finished because the output tokens exceed max_new_tokens, meaning that the request is truncated
    def add_sampled_token(self, tok: int, eos_token_id_set, now) -> int:
        num_truncated: int = 0
        if tok in eos_token_id_set \
            or len(self.output_ids)+1 >= self.max_new_tokens:
            self.finished = True

            if len(self.output_ids)+1 >= self.max_new_tokens:
                num_truncated += 1

            # self.io_token_ids, do not append when finished
            self.output_ids.append(tok) if tok not in eos_token_id_set else None
            self.metrics.report(now)     # regardless of EOS or not

            if self.loop is not None:
                self.loop.call_soon_threadsafe(self.token_queue.put_nowait, tok) if tok not in eos_token_id_set else None
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
    slots: list[int]

    def __post_init__(self):
        assert self.want == len(self.slots)

class SchedulerOutput:
    def __init__(self, step: int, reqs: list[ModelRequest], scheduled: dict[str, ScheduledInfo],
                 block_tables: list[list[int]], config: ModelConfig, scheduler: "Scheduler | None"):
        assert len(reqs) > 0, "SchedulerOutput must have at least one request"
        self.reqs = reqs
        self.scheduled = scheduled
        self.block_tables = block_tables
        self.config = config

        self.batch_size = len(reqs)
        self.prompt_ids = [req.input_ids for req in reqs]
        batch_sampling = [req.sampling for req in reqs]     # list[Sampling | None]
        self.tensor_sampling = TensorSampling.from_sampling_list(batch_sampling, self.config, self.batch_size)

        # intermediate states for the batch
        self.output_ids: list[list[int]] = [req.output_ids for req in self.reqs]      # for repetition penalty and synchronous generation
        self.finished: list[bool] = [req.finished for req in self.reqs]

        # metrics
        num_prefill_tokens, num_decode_tokens = 0, 0
        for req in self.reqs:
            if req.metrics.first_schedule_time is None:
                req.metrics.first_schedule_time = time.perf_counter()
            if not req.is_decoding:
                # prefill
                num_prefill_tokens += scheduled[req.request_id].want
                req.metrics.num_prefill_chunk += 1
            else:
                # decode
                num_decode_tokens += scheduled[req.request_id].want

        self.step_metrics: SchedulerStepMetrices | None = SchedulerStepMetrices(
            step=step,
            bz=self.batch_size,
            n_p=num_prefill_tokens,
            n_d=num_decode_tokens,
            run=len(scheduler.running),
            wait=len(scheduler.waiting),
            blk_used=scheduler.cache.pool.used(),
            blk_total=scheduler.cache.pool.num_blocks,
        ) if scheduler is not None else None

    # return the number of truncated reqs which are finished just now
    def add_sampled_tokens(self, next_tokens_cpu: list[int], eos_token_id_set) -> int:
        assert len(next_tokens_cpu) == self.batch_size
        num_truncated: int = 0

        now = time.perf_counter()
        for (tok, req) in zip(next_tokens_cpu, self.reqs):
            assert not req.finished

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"SchedulerOutput, req_id: {req.request_id}, before add, len: {len(req.output_ids)}, max: {req.max_new_tokens}, sampled token={tok}")

            s_info = self.scheduled[req.request_id]

            # chunked prefill, do not add tok until all prefill-needed tokens are done.
            if not req.is_decoding:
                req.num_computed_tokens += s_info.want
                assert req.num_computed_tokens <= req._num_prefill_tokens
                if req.num_computed_tokens == req._num_prefill_tokens:
                    req.is_decoding = True
                    num_truncated += req.add_sampled_token(tok, eos_token_id_set, now)
                continue

            # in decoding
            req.num_computed_tokens += s_info.want
            num_truncated += req.add_sampled_token(tok, eos_token_id_set, now)

        return num_truncated

class Scheduler:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.max_waiting = config.max_waiting
        self.long_prefill_token_threshold = config.long_prefill_token_threshold
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_num_seqs = config.max_num_seqs
        assert self.max_num_batched_tokens >= self.max_num_seqs # ensure that all reqs in decoding can be admitted.

        # scheduling strategy
        self.use_d_first_schedule = config.use_d_first_schedule

        self.waiting: deque[ModelRequest] = deque()
        self.running: list[ModelRequest] = []
        self.cache = KVCache(config)

        # backoff after preempted
        self.backoff_base = config.backoff_base
        self.backoff_cap = config.backoff_cap

        # metrics
        self.sch_metrics = SchedulerMetrices()
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
        max_blocks = cdiv(len(req.input_ids) + req.max_new_tokens, self.cache.block_size)
        if max_blocks > self.cache.pool.num_blocks:
            return False        # can never be served; reject at admission, not at allocation
        
        if len(self.waiting) >= self.max_waiting:
            return False
        
        self.waiting.append(req)
        return True

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def _preempt(self, victim: ModelRequest, victims: set[str]):
        '''
        todo: swap out to host memory
        new issue: how to choose between swap out and recompute
        '''
        assert not victim.finished

        self.sch_metrics.report_on_preemption()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{victim.request_id} preempted, moved from running to waiting")

        victims.add(victim.request_id)
        self.running.remove(victim) if victim in self.running else None   # compatible for D_first_preemptive_schedule
        self.cache.free(victim)     # must execute before reset_on_preemption because the original num_computed_tokens is needed for freeing cache

        victim.reset_on_preemption()  # recompute from scratch, preempt_count += 1

        delay = min(self.backoff_base ** (victim.preempt_count - 1), self.backoff_cap)  # mininum of delay is 1
        victim.not_before_step = self.sch_metrics.step + delay
        self.waiting.appendleft(victim)

    def commit_step(self, sch_out: SchedulerOutput, num_truncated:int):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"call commit_step, ")
        for req in sch_out.reqs:
            if req.finished:
                self.cleanup_on_finished(req=req)

        self.sch_metrics.report_on_truncated(num_truncated)
        self.log_metrics(sch_out=sch_out)

    def cleanup_on_finished(self, req: ModelRequest):
        assert req.finished and req.is_decoding
        assert req in self.running

        self.sch_metrics.report_on_finish()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{req.request_id} finished, removed")

        self.running.remove(req)
        self.cache.free(req)
        self.req_metrics_list.append(req.metrics)

    def _do_cleanup_on_error(self, req: ModelRequest, e: Exception):
        req.finished = True
        if req.loop is not None:
            req.loop.call_soon_threadsafe(req.token_queue.put_nowait, e)

        self.cache.free(req)
        self.req_metrics_list.append(req.metrics)
        self.sch_metrics.report_on_error()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"an error occured in {req.request_id}, removed")

    def cleanup_running_on_error(self, e: Exception):
        while self.running:
            req = self.running.pop()
            self._do_cleanup_on_error(req, e)

        # maybe cuased by the front of waiting
        req = self.waiting.popleft()
        self._do_cleanup_on_error(req, e)

    def cleanup_on_error(self, req: ModelRequest, e: Exception):
        assert req in self.running
        self.running.remove(req)

        self._do_cleanup_on_error(req, e)

    def cleanup_on_abort(self, request_id: str):
        '''called from api.py maybe due to the connection lost, but the request may have finished now'''
        req = None
        for i in range(len(self.running)):
            if self.running[i].request_id == request_id:
                req = self.running.pop(i)

        if req is None:
            for i in range(len(self.waiting)):
                if self.waiting[i].request_id == request_id:
                    # delete at index from deque
                    self.waiting.rotate(-i)
                    req = self.waiting.popleft()
                    self.waiting.rotate(i)

        if req is None:
            '''do not exist in both running and waiting, may have finished, so do nothing'''
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"the aborted request may have finished, do nothing.")
            return

        self.cache.free(req)
        self.req_metrics_list.append(req.metrics)
        self.sch_metrics.report_on_error()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{request_id} aborted, removed")

    def _pick_waiting(self, victims: set[str]) -> ModelRequest | None:
        '''
        todo:
        1) cdiv(len(input_ids) + expected_output_len, block_size) to avoid future preemption, 
              in which expected_output_len is evaluated by 50p of historical output lengths
        2) aging boost to avoid long prefills' starvation.
        '''
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"len, victims: {len(victims)}, waiting: {len(self.waiting)}")

        for r in self.waiting:
            if r.request_id not in victims and r.not_before_step <= self.sch_metrics.step:
                return r

        # ignore backoff when len(running)==0 and len(waiting)>0 and nothing available due to backoff, avoiding the starvation of scheduler.
        # victims must be empty in this situation.
        # expect: this optimizaiton only gains a little benefit because run_loop will continue when the current function _pick_waiting returns None.
        if len(self.running)==0 and len(self.waiting)>0:
            return self.waiting[0]      # oldest one

        return None

    def schedule(self) -> SchedulerOutput | None:            # called by run_loop
        self.sch_metrics.step += 1
        return self.D_first_preemptive_schedule() if self.use_d_first_schedule else self.preemptive_schedule()

    # D-first scheduling with no-cross preemption
    def D_first_preemptive_schedule(self) -> SchedulerOutput | None:            # called by run_loop when idle
        budget = self.max_num_batched_tokens
        scheduled: dict[str, ScheduledInfo] = {}
        victims: set[str] = set()

        # pre-process: ensure that all Ds are before all Ps
        decoding, prefill = [], []
        for req in list(self.running):
            (decoding if req.is_decoding else prefill).append(req)
        self.running: list[ModelRequest] = decoding + prefill

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"len, running: {len(self.running)}, waiting: {len(self.waiting)}")

        # 1) running first — protect in-flight decodes' TPOT.
        scheduled_running: list[ModelRequest] = []
        for req in list(self.running):
            assert not req.finished     # all finished requests has been removed by cleanup_on_finished in commit_step after add_sampled_tokens

            if req.request_id in victims:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"{req.request_id} has been prempted")
                break
            
            want = 1 if req.is_decoding else min(req.num_prompt_remaining, budget, self.long_prefill_token_threshold)
            if want <= 0:
                # only when in prefill (is_decoding=False) and budget <= 0, which means all Ds has been scheduled and budget exhausted, the loop breaks.
                # then kept the rest in running, but not be scheduled
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"no more budget for {req.request_id} to prefill, skip over")
                break

            new_slots = self.cache.allocate_slots(req, want)
            while new_slots is None:       # KV pool exhausted
                self.sch_metrics.report_on_cache_exhausted()

                victim = self.running.pop()     # traverse reversely, so it may have finished.
                self._preempt(victim, victims)     # yield no matter whether it's in decoding

                if req.request_id == victim.request_id:     # cur req preempted
                    break

                new_slots = self.cache.allocate_slots(req, want)

            if new_slots is None:               # still cannot get new blocks
                break

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"add {req.request_id} to scheduled_running")
            scheduled_running.append(req)
            s_info = ScheduledInfo(want=want, slots=new_slots)
            scheduled[req.request_id] = s_info
            budget -= want
    
        # 2) waiting next — fill remaining budget with (chunked) prefills
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            req = self._pick_waiting(victims=victims)
            if req is None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"not find a suitable waiting request, waiting len: {len(self.waiting)}, victims len: {len(victims)}")
                break

            want = min(req.num_prompt_remaining, budget, self.long_prefill_token_threshold)
            slots = self.cache.allocate_slots(req, want=want, respect_watermark=True)
            if slots is None:
                self.sch_metrics.report_on_cache_exhausted()
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"failed to allocate slots for {req.request_id}")
                break                                 # no room, stop admitting

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"move {req.request_id} from waiting to running and scheduled_running")
            scheduled_running.append(req)
            self.running.append(req)
            self.waiting.remove(req)
            s_info = ScheduledInfo(want=want, slots=slots)
            scheduled[req.request_id] = s_info
            budget -= want

        block_tables = []
        for req in scheduled_running:
            bt = self.cache.get_block_table(req)
            assert bt is not None
            block_tables.append(bt)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"schedule1, scheduled: {len(scheduled_running)}, running: {len(self.running)}, waiting: {len(self.waiting)}")

        if not scheduled_running:
            return None     # no work to do

        self.sch_metrics.report_on_schedule(scheduled_reqs=scheduled_running)
        return SchedulerOutput(self.sch_metrics.step, scheduled_running, scheduled, block_tables=block_tables,
                               config=self.config, scheduler=self)

    def _pick_victim(self, cur_req):
        # pick strategies
        # 1) if cur_req is D, any P can be preempted. pick the newest D when there is no P.
        # 2) if cur_req is P, only Ps after cur_req can be preempted.
        for req in reversed(self.running):
            if not cur_req.is_decoding and req.request_id == cur_req.request_id:  # strategy 2
                    return None

            if not req.is_decoding:     # strategy 1
                return req

        # no P available: cur_req must be D, then apply strategy 1
        newest = self.running[-1]       # the last one, that is, the newest one.
        return None if newest.request_id == cur_req.request_id else newest

    # Preemptive scheduling with victim eviction
    def preemptive_schedule(self) -> SchedulerOutput | None:            # called by run_loop when idle
        budget = self.max_num_batched_tokens
        scheduled: dict[str, ScheduledInfo] = {}
        victims: set[str] = set()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"len, running: {len(self.running)}, waiting: {len(self.waiting)}")
        
        # 1) running first — protect in-flight decodes' TPOT.
        #    Note: Ps and Ds may interleave.
        scheduled_running: list[ModelRequest] = []
        for req in list(self.running):  # snapshot
            assert not req.finished     # all finished requests has been removed by cleanup_on_finished in commit_step after add_sampled_tokens

            if req.request_id in victims:    # evicted
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"{req.request_id} has been prempted")
                continue

            want = 1 if req.is_decoding else min(req.num_prompt_remaining, budget, self.long_prefill_token_threshold)
            if want <= 0:
                # only when is_decoding=False and budget <= 0, which means there is no room for current in-flight prefill request.
                # then kept the rest in running, but not be scheduled
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"no more budget for {req.request_id} to prefill, skip over")
                continue

            new_slots = self.cache.allocate_slots(req, want)
            while new_slots is None:                # KV pool exhausted
                self.sch_metrics.report_on_cache_exhausted()

                victim = self._pick_victim(req)
                if victim is None:
                    self._preempt(req, victims)     # yield no matter whether it's in decoding
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"{req.request_id} preempts itself")
                    break

                self._preempt(victim, victims)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"{req.request_id} preempts {victim.request_id}")

                # roll back budget and blocks just allocated
                s_info = scheduled.pop(victim.request_id, None)
                if s_info is not None:
                    scheduled_running.remove(victim)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"remove {req.request_id} from scheduled_running")
                    budget += s_info.want

                new_slots = self.cache.allocate_slots(req, want)

            if new_slots is None:                   # yield, according to "victim is None"
                continue

            s_info = ScheduledInfo(want=want, slots=new_slots)
            scheduled[req.request_id] = s_info
            scheduled_running.append(req)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"add {req.request_id} to scheduled_running")
            budget -= want
    
        # 2) waiting next — fill remaining budget with (chunked) prefills
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            req = self._pick_waiting(victims=victims)
            if req is None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"not find a suitable waiting request, waiting len: {len(self.waiting)}, victims len: {len(victims)}")
                break

            want = min(req.num_prompt_remaining, budget, self.long_prefill_token_threshold)
            slots = self.cache.allocate_slots(req, want, respect_watermark=True)
            if slots is None:
                self.sch_metrics.report_on_cache_exhausted()
                break                                 # no room, stop admitting

            self.running.append(req)
            self.waiting.remove(req)
            s_info = ScheduledInfo(want=want, slots=slots)
            scheduled[req.request_id] = s_info
            scheduled_running.append(req)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"add {req.request_id} to scheduled_running")
            budget -= want

        if not scheduled_running:
            return None     # no work to do

        block_tables = []
        for req in scheduled_running:
            bt = self.cache.get_block_table(req)
            assert bt is not None
            block_tables.append(bt)

        self.sch_metrics.report_on_schedule(scheduled_reqs=scheduled_running)
        return SchedulerOutput(self.sch_metrics.step, scheduled_running, scheduled, block_tables=block_tables,
                               config=self.config, scheduler=self)

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
        tmp_counters: SchedulerMetrices = copy.deepcopy(self.sch_metrics)
        req_metrics_list = self.req_metrics_list
        self.req_metrics_list = []      # reset
        self.total_metrics.extend(req_metrics_list) if self.is_benchmarking else None

        if logger.isEnabledFor(logging.DEBUG):
            for metrics in req_metrics_list:
                json_bytes = orjson.dumps(round_floats(asdict(metrics), nd=3))
                logger.debug(f"metrics obj: {json_bytes.decode()}")

        # analyze
        json_bytes = analyze_metrics(req_metrics_list=req_metrics_list, sch_metrics=tmp_counters,
                                     is_benchmarking=self.is_benchmarking, config=self.config)
        logger.info(f"analyzed scheduler metrics: {json_bytes.decode()}")

        # save to file
        log_path = Path(self.config.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        stats_path = log_path / f'sch_metrics.{self.log_name_flag}.json'
        with open(stats_path, "a+b") as f:
            f.write(json_bytes)
            f.write(b"\n")
            f.flush()

    def log_step_metrics(self, sch_out: SchedulerOutput):
        assert sch_out.step_metrics
        json_bytes = orjson.dumps(round_floats(asdict(sch_out.step_metrics), nd=3))

        # save to file
        log_path = Path(self.config.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        stats_path = log_path / f'step_metrics.{self.log_name_flag}.json'
        with open(stats_path, "a+b") as f:
            f.write(json_bytes)
            f.write(b"\n")
            f.flush()

