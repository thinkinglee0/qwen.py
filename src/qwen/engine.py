import gc
import logging
import threading
import time
import torch
import asyncio
from typing import AsyncIterator

from qwen.metrics import SchedulerStepMetrics, timed
from qwen.model import QwenForCausalLM
from qwen.attention import build_attn_metadata
from qwen.sampling import SamplingParams
from qwen.scheduler import Scheduler, SchedulerOutput, ModelRequest
from qwen.constants import DEFAULT_MAX_NEW_TOKEN
from qwen.config import ModelConfig

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, config: ModelConfig):
        # Warning: do not change the construction order of QwenForCausalLM and Scheduler
        # load weights. its construction must be before kv cache' because of kv cache's available memory check. 
        self.model = QwenForCausalLM(config=config)

        # allocate kv cache in its constructor, and check whether available gpu memory is enough for the specified num_blocks.
        # see KVCacheData for details.
        # Warning: must use model's config snapshot to initialize Scheduler, ensure that they share the same config.
        self.scheduler = Scheduler(config=self.model.config)

    def step(self) -> bool:
        step_metrics = SchedulerStepMetrics()
        step_metrics.start("step")
        sch_out = self.scheduler.schedule(step_metrics=step_metrics)
        if sch_out is None:
            logger.info("Scheduler: no work, next loop iteration")
            return False

        self.forward(sch_out=sch_out)
        return True


    @torch.inference_mode()
    def forward(self, sch_out: SchedulerOutput):
        assert sch_out.step_metrics is not None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"batch size: {len(sch_out.reqs)}")

        with timed(sch_out.step_metrics, "bld_meta"):
            packed_input_ids, md = build_attn_metadata(
                sch_out,
                cache_data=self.scheduler.cache.data,
                config=self.model.config,
                rope=self.model.model.rope)

        with timed(sch_out.step_metrics, "fwd"):
            # torch.cuda.set_sync_debug_mode("error")
            hidden = self.model.forward(packed_input_ids, md)       # [total_tokens, H]
            # torch.cuda.set_sync_debug_mode("default")

        with timed(sch_out.step_metrics, "logits"):
            # gather each seq's LAST token -> logits -> first generated token
            last_idx = md.cu_seqlens_q[1:] - 1          # [B]
            logits = self.model.compute_logits(hidden[last_idx])   # [B, vocab]

        with timed(sch_out.step_metrics, "sample"):
            next_tokens = self.model.sampler(logits, sch_out)          # [B]

        with timed(sch_out.step_metrics, "dth"):
            next_tokens_cpu = next_tokens.tolist()

        sch_out.step_metrics.n_sample = len(next_tokens_cpu)

        # merge md.step_metrics_lst to sch_out.step_metrics
        # must be placed after next_tokens.tolist(), a synchronous operation, otherwise all events would be not ready.
        if md.step_metrics_lst:
            for layer_metrics in md.step_metrics_lst:
                sch_out.step_metrics.merge(layer_metrics)

        sch_out.step_metrics.start("ci")
        num_truncated = sch_out.add_sampled_tokens(next_tokens_cpu, self.model.config.eos_token_id_set)

        self.scheduler.commit_step(sch_out=sch_out, num_truncated=num_truncated)


    def run_to_completion(self):
        while self.scheduler.has_unfinished():
            try:
                if not self.step():
                    continue
            except Exception as e:
                logger.exception(f"Error occurred while generating")
                break

    def teardown(self):
        self.scheduler.teardown()
        del self.scheduler

def benchmark(engine: LLMEngine, batch_input_ids: list[list[int]],
              sampling: SamplingParams | None = None, max_new_tokens: int = DEFAULT_MAX_NEW_TOKEN) -> tuple[list[list[int]], float]:
    """Run all requests to completion without blocking, for benchmarking."""
    assert len(batch_input_ids) > 0 and len(batch_input_ids) <= engine.scheduler.max_waiting, "batch_input_ids must not be empty or longer than max_waiting queue"
    logger.info(f"request count: {len(batch_input_ids)}")

    reqs: list[ModelRequest] = []
    for input_ids in batch_input_ids:
        loop = None
        req = ModelRequest(engine.model.config, loop, input_ids=input_ids, sampling=sampling, max_new_tokens=max_new_tokens)
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

    def submit(self, input_ids, request_id: str | None, sampling, max_new_tokens: int)  -> asyncio.Queue[int | None | Exception] | None:
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
        with self.cond:
            self.engine.scheduler.cleanup_on_abort(request_id)

    def run_loop(self):
        while not self._shutdown:
            scheduler_output = None     # avoid UnboundLocalError if schedule() raised
            try:
                with self.cond:
                    while not self.engine.scheduler.has_unfinished() and not self._shutdown:
                        self.cond.wait()

                    if self._shutdown:  # after awaken
                        break

                    # may return None when the waiting is not empty due to backoff
                    step_metrics = SchedulerStepMetrics()
                    step_metrics.start("step")
                    scheduler_output = self.engine.scheduler.schedule(step_metrics=step_metrics)

                if not scheduler_output:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Scheduler: no work due to backoff or empty waiting, next loop iteration")
                    continue

                self.engine.forward(sch_out=scheduler_output)
            except Exception as e:
                logger.exception("Error occurred in schedule or forward")
                self._cleanup_on_error(scheduler_output, e)

    def _cleanup_on_error(self, scheduler_output: SchedulerOutput | None, e: Exception):
        """Best-effort cleanup after a failed step: never let it raise out of run_loop."""
        if scheduler_output is None:
            # the error occurred in schedule(), so there is no batch to blame: drop the running queue
            try:
                self.engine.scheduler.cleanup_all_on_error(e=e)
            except Exception:
                logger.exception("error while cleaning up the running queue")
            return

        # keep going over the whole batch even if one request fails to clean up
        for req in scheduler_output.reqs:
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



