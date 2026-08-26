import logging
import threading
import time
import torch
import asyncio
from typing import AsyncIterator

from qwen.model import QwenForCausalLM
from qwen.cache import KVCache, KVCacheData
from qwen.attention import build_attn_metadata
from qwen.sampling import Sampling
from qwen.scheduler import Scheduler, SchedulerOutput, ModelRequest
from qwen.constants import DEFAULT_MAX_NEW_TOKEN
from qwen.config import ModelConfig

logger = logging.getLogger(__name__)


@torch.inference_mode()
def _generate(
    model: QwenForCausalLM,
    cache_data: KVCacheData,
    sch_out: SchedulerOutput,
    scheduler: Scheduler,
):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"batch size: {len(sch_out.reqs)}")

    packed_input_ids, md = build_attn_metadata(sch_out, cache_data, model.config.device)  # is_prefill=True
    start_time = time.perf_counter()
    hidden = model.forward(packed_input_ids, md)       # [total_tokens, H]
    assert sch_out.step_metrics is not None
    sch_out.step_metrics.elapsed_ms = (time.perf_counter() - start_time) * 1000

    # gather each seq's LAST token -> logits -> first generated token
    last_idx = md.cu_seqlens_q[1:] - 1          # [B]
    logits = model.compute_logits(hidden[last_idx])   # [B, vocab]
    next_tokens = model.sampler(logits, sch_out)          # [B]

    num_truncated = sch_out.add_sampled_tokens(next_tokens.tolist(), model.config.eos_token_id_set)

    scheduler.commit_step(sch_out=sch_out, num_truncated=num_truncated)

class LLMEngine:
    def __init__(self, config: ModelConfig):
        # Warning: do not change the construction order of QwenForCausalLM and Scheduler
        # load weights. its construction must be before kv cache' because of kv cache's available memory check. 
        self.model = QwenForCausalLM(config=config)

        # allocate kv cache in its constructor, and check whether available gpu memory is enough for the specified num_blocks.
        # see KVCacheData for details.
        # Warning: must use model's config snapshot to initialize Scheduler, ensure that they share the same config.
        self.scheduler = Scheduler(config=self.model.config)

    def run_to_completion(self):
        while self.scheduler.has_unfinished():
            scheduler_output = self.scheduler.schedule()
            if scheduler_output is None:
                logger.info("Scheduler: no work, next loop iteration")
                continue
            try:
                _generate(model=self.model, cache_data=self.scheduler.cache.data,
                          sch_out=scheduler_output, scheduler=self.scheduler)
            except Exception as e:
                logger.exception(f"Error occurred while generating")
                break

    def teardown(self):
        self.scheduler.teardown()
        del self.scheduler

def benchmark(engine: LLMEngine, batch_input_ids: list[list[int]],
              sampling: Sampling | None = None, max_new_tokens: int = 300) -> tuple[list[list[int]], float]:
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
                    scheduler_output = self.engine.scheduler.schedule()

                if not scheduler_output:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Scheduler: no work due to backoff or empty waiting, next loop iteration")
                    continue

                _generate(model=self.engine.model, cache_data=self.engine.scheduler.cache.data,
                          sch_out=scheduler_output, scheduler=self.engine.scheduler)
            except Exception as e:
                logger.exception("Error occurred in schedule or _generate")
                try:
                    if scheduler_output is not None:
                        for req in scheduler_output.reqs:
                            self.engine.scheduler.cleanup_on_error(req=req, e=e)
                    else:
                        # error occured in schedule, clean up the running queue
                        self.engine.scheduler.cleanup_running_on_error(e=e)

                except RuntimeError as e:
                    logger.exception(f"runtime error when clean up {req.request_id}")

async def async_generate(
    driver: ServingDriver,
    input_ids: list[int],    # one variable-length id sequence
    request_id: str | None = None,
    sampling: Sampling | None = None,
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



