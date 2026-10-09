import gc
import logging
import threading
import time
import torch
import asyncio
from typing import AsyncIterator
from collections import deque

from qwen.metrics import SchedulerStepMetrics, timed
from qwen.model import QwenForCausalLM
from qwen.sampling import SamplingParams
from qwen.scheduler import Scheduler, SchedulerOutput, ModelRequest
from qwen.constants import DEFAULT_MAX_NEW_TOKEN
from qwen.config import ModelConfig

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, config: ModelConfig):
        # WARNING: do not change the construction order of QwenForCausalLM and Scheduler
        # load weights. its construction must be before kv cache' because of kv cache's available memory check. 
        self.model = QwenForCausalLM(config=config)

        # allocate kv cache in its constructor, and check whether available gpu memory is enough for the specified num_blocks.
        # see KVCacheData for details.
        # WARNING: must use model's config snapshot to initialize Scheduler, ensure that they share the same config.
        self.scheduler = Scheduler(config=self.model.config)

        # steps that has been launched, but not land.
        self.in_flight_steps: deque[SchedulerOutput] = deque()

    def step(self) -> bool:
        step_metrics = SchedulerStepMetrics()
        with timed(step_metrics, "step_0"):
            sch_out = self.scheduler.schedule(step_metrics=step_metrics)
            bsz = self.forward(sch_out=sch_out)

        s_p = self.handle_in_flight_step(cur_sch_out=sch_out)
        if not bsz and not s_p:
            logger.info(f"no request handled, step_id: {self.scheduler.sch_metrics.step_id}")
        
        return self.has_unfinished()

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished() or bool(self.in_flight_steps)
    
    @torch.inference_mode()
    def forward(self, sch_out: SchedulerOutput | None) -> int:
        if sch_out is None:
            logger.info(f"no available request in running or waiting, step_id: {self.scheduler.sch_metrics.step_id}")
            return 0
                
        assert sch_out.step_metrics is not None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"batch size: {sch_out.batch_size}, step_id: {self.scheduler.sch_metrics.step_id}")

        with timed(sch_out.step_metrics, "bld_meta"):
            packed_input_ids, md = sch_out.build_attn_metadata(
                cache_data=self.scheduler.cache.data,
                rope=self.model.model.rope)

        with timed(sch_out.step_metrics, "fwd"):
            hidden = self.model.forward(packed_input_ids, md)       # [total_tokens, H]

        with timed(sch_out.step_metrics, "logits"):
            # gather each seq's LAST token -> logits -> first generated token
            last_idx = md.cu_seqlens_q[1:] - 1          # [B]
            logits = self.model.compute_logits(hidden[last_idx])   # [B, vocab]

        with timed(sch_out.step_metrics, "smp"):
            with timed(sch_out.step_metrics, "smp_prep"):
                sampling_tensors = sch_out.build_sampling_tensors()

            with timed(sch_out.step_metrics, "smp_run"):
                next_tokens = self.model.sampler(logits, sampling_tensors, sch_out.step_metrics)          # [B]

            with timed(sch_out.step_metrics, "smp_post"):
                assert sch_out.slot_idx is not None and sch_out.want is not None and sch_out.needs_sample_device is not None
                sch_out.dth_buf = self.scheduler.tok_id_tab.add_sampled_tokens_on_device(
                    slot_idx=sch_out.slot_idx, needs_sample=sch_out.needs_sample_device,
                    next_tokens=next_tokens, want=sch_out.want,
                    step_metrics=sch_out.step_metrics)

        with timed(sch_out.step_metrics, "upd_proj"):
            sch_out.update_projected_state_in_advance()
            self.in_flight_steps.append(sch_out)
            sch_out.incr_num_in_flight()
        return sch_out.batch_size

    def pop_landable_step(self, cur_sch_out: SchedulerOutput | None) -> SchedulerOutput | None:
        if not self.in_flight_steps:
            logger.info(f"no in-flight request, step_id: {self.scheduler.sch_metrics.step_id}")
            return None

        if cur_sch_out is self.in_flight_steps[0]:
            return None     # the first step is still in-flight, do nothing

        pre_sch_out: SchedulerOutput = self.in_flight_steps.popleft()
        assert pre_sch_out is not None

        return pre_sch_out

    def sync_dth(self, pre_sch_out: SchedulerOutput):
        assert pre_sch_out.step_metrics is not None

        with timed(pre_sch_out.step_metrics, "dth_wait"):
            pre_sch_out.step_metrics.events.synchronize("dth")  # wait for data from device.

    # NOTE: need lock in run_loop to protect running/waiting queue, 
    # because land_step() will call scheduler.commit_step() which modifies the running/waiting queue.
    def land_step(self, pre_sch_out: SchedulerOutput) -> int:
        assert pre_sch_out.step_metrics is not None
        with timed(pre_sch_out.step_metrics, "step_1"):
            num_truncated = self.land_step_imp(pre_sch_out=pre_sch_out)

            with timed(pre_sch_out.step_metrics, "ci"):
                self.scheduler.commit_step(sch_out=pre_sch_out, num_truncated=num_truncated)

        # NOTE: the cpu time of log_metrics not taken into account by 'step_1'
        self.scheduler.log_metrics(sch_out=pre_sch_out)

        return pre_sch_out.batch_size

    def land_step_imp(self, pre_sch_out: SchedulerOutput) -> int:
        assert pre_sch_out.step_metrics is not None
        pre_sch_out.step_metrics.pend = len(self.in_flight_steps)     # a value after this step

        # merge md.step_metrics_lst to pre_sch_out.step_metrics
        # must be placed after sync_dth(), a synchronous operation, otherwise all events would be not ready.
        md = pre_sch_out.attn_meta
        assert md is not None
        if md.step_metrics_lst:
            for layer_metrics in md.step_metrics_lst:
                pre_sch_out.step_metrics.merge(layer_metrics)

        num_truncated = pre_sch_out.add_sampled_tokens_on_host()

        return num_truncated

    def handle_in_flight_step(self, cur_sch_out: SchedulerOutput | None) -> int:
        pre_sch_out = self.pop_landable_step(cur_sch_out=cur_sch_out)
        if pre_sch_out is None:
            return 0
        
        self.sync_dth(pre_sch_out=pre_sch_out)

        return self.land_step(pre_sch_out=pre_sch_out)

    def run_to_completion(self):
        while self.has_unfinished():
            try:
                if not self.step():
                    break
            except Exception as e:
                logger.exception(f"Error occurred while generating")
                break

    def teardown(self):
        self.scheduler.teardown()
        del self.scheduler

def benchmark(engine: LLMEngine, batch_input_ids: list[list[int]],
              sampling: SamplingParams | None = None, max_new_tokens: int = DEFAULT_MAX_NEW_TOKEN,
              specify_request_id: bool = False,
              ) -> tuple[list[list[int]], float]:
    """Run all requests to completion without blocking, for benchmarking."""
    assert len(batch_input_ids) > 0 and len(batch_input_ids) <= engine.scheduler.max_waiting, "batch_input_ids must not be empty or longer than max_waiting queue"
    logger.info(f"request count: {len(batch_input_ids)}")

    reqs: list[ModelRequest] = []
    for i, input_ids in enumerate(batch_input_ids):
        loop = None
        request_id = f"req_{i}" if specify_request_id else None
        req = ModelRequest(engine.model.config, loop, input_ids=input_ids, sampling=sampling, max_new_tokens=max_new_tokens, request_id=request_id)
        if engine.scheduler.add_request(req):       # very unlikely to reject in this test scenario
            reqs.append(req)
        else:
            raise RuntimeError("Request rejected: too many waiting requests")

    t0 = time.perf_counter()
    engine.run_to_completion()         # drains the whole queue synchronously
    elapsed = time.perf_counter() - t0

    engine.scheduler.log_scheduler_metrics(is_exiting=True)    # for last metrics but the logging interval does not elapse.

    output_ids = [req.output_ids for req in reqs]
    return output_ids, elapsed


class ServingDriver:
    def __init__(self, engine: LLMEngine):
        self._shutdown = False
        self.engine = engine
        self.lock = threading.Lock()          # lock lives HERE, not in scheduler
        self.cond = threading.Condition(self.lock)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def start(self):
        logger.info("run_loop thread is starting")
        self._thread = threading.Thread(target=self.run_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float=5.):
        time.sleep(timeout)
        self._shutdown = True

        self.engine.teardown()
        del self.engine

        if torch.cuda.is_available():
            gc.collect()
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def submit(self, input_ids: list[int], request_id: str | None, sampling, max_new_tokens: int)  -> asyncio.Queue[int | None | Exception] | None:
        with self.cond:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"submit: input_ids: {input_ids}, sampling: {sampling}")
            loop = asyncio.get_running_loop()
            req = ModelRequest(self.engine.model.config, loop, input_ids=input_ids,
                               request_id=request_id, sampling=sampling, max_new_tokens=max_new_tokens)
            if self.engine.scheduler.add_request(req):
                self.cond.notify()
                return req.token_queue
            else:
                return None

    def abort(self, request_id: str):
        with self.cond:     # protect running/waiting queue
            self.engine.scheduler.cleanup_on_abort(request_id)

    def run_loop(self):
        while not self._shutdown:
            sch_out = None     # avoid UnboundLocalError if schedule() raised
            pre_sch_out = None
            try:
                step_metrics = SchedulerStepMetrics()
                with self.cond:
                    while not self.engine.has_unfinished() and not self._shutdown:
                        self.cond.wait()

                    if self._shutdown:  # after awaken
                        break

                    # may return None when the waiting is not empty due to backoff
                    step_metrics.start("step_0")
                    sch_out = self.engine.scheduler.schedule(step_metrics=step_metrics)

                # out of the lock
                self.engine.forward(sch_out=sch_out)
                step_metrics.stop("step_0")

                # NOTE: must call pop_landable_step() and land_step() in the same thread as forward(),
                # otherwise the dth_buf may be freed before land_step() is called.
                pre_sch_out = self.engine.pop_landable_step(cur_sch_out=sch_out)
                if pre_sch_out is None:
                    continue

                self.engine.sync_dth(pre_sch_out=pre_sch_out)
                with self.cond: # protect running/waiting queue
                    self.engine.land_step(pre_sch_out=pre_sch_out)
            except Exception as e:
                logger.exception("Error occurred in schedule/forward/land")
                with self.cond: # protect running/waiting queue
                    # pre_sch_out is not None: pop_landable_step() succeeded, but land_step() failed
                    # sch_out is not None: schedule() or forward() failed
                    # NOTE: must check pre_sch_out first, because it's not None meaning that sch_out has been generated and handled successfully.
                    err_sch_out = pre_sch_out if pre_sch_out is not None else sch_out
                    if err_sch_out is not None:
                        self._cleanup_on_error(err_sch_out, e)
                    else:
                        self._cleanup_all_on_error(e)

    def _cleanup_all_on_error(self, e: Exception):
        # the error occurred in schedule(), so there is no batch to blame: 
        # drop the running/waiting queue
        try:
            self.engine.scheduler.cleanup_all_on_error(e=e)
        except Exception:
            logger.exception("error while cleaning up the running/waiting queue")
        return

    def _cleanup_on_error(self, sch_out: SchedulerOutput, e: Exception):
        # keep going over the whole batch even if one request fails to clean up
        assert sch_out is not None
        for req in sch_out.reqs:
            try:
                self.engine.scheduler.cleanup_on_error(req=req, e=e)
            except Exception:
                logger.exception(f"error while cleaning up {req.request_id}")

async def async_generate(
    driver: ServingDriver,
    input_ids: list[int],    # one variable-length id sequence
    request_id: str | None = None,
    sampling: SamplingParams | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKEN,
) -> AsyncIterator[int]:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"async_generate: input_ids: {input_ids}, sampling: {sampling}")
    queue = driver.submit(input_ids, request_id, sampling, max_new_tokens)

    if queue is None:
        raise RuntimeError("Request rejected: too many waiting requests")

    # output input_ids
    for tok in input_ids:
        yield tok

    # output generated tokens
    while True:
        tok = await queue.get()                # suspends coroutine, frees event loop
        if tok is None:                              # sentinel = stream end
            break
        elif isinstance(tok, Exception):
            raise tok
        else:
            yield tok



