from collections import deque
import logging
import copy
import torch
import asyncio
import time
from datetime import datetime
from pathlib import Path

from qwen.sampling import SamplingMetadata
from qwen.config import ModelConfig
from qwen.metrics import analyze_stats, Metrics
from qwen.constants import LOG_DIR

logger = logging.getLogger(__name__)


class ModelRequest:
    def __init__(self, loop, input_ids: list[int], sampling, max_new_tokens):
        self.input_ids = input_ids
        self.sampling = sampling
        self.max_new_tokens = max_new_tokens
        self.loop = loop
        self.token_queue: asyncio.Queue[int | None | Exception] = asyncio.Queue()

        # metrics
        self.arrival_time = time.perf_counter()
        

class BatchRequest:
    def __init__(self, reqs: list[ModelRequest], config, scheduler: "Scheduler | None" = None):
        assert len(reqs) > 0, "BatchRequest must have at least one request"
        self.reqs = reqs
        self.scheduler = scheduler

        self.batch_size = len(reqs)
        self.prompt_ids = [req.input_ids for req in reqs]
        batch_sampling = [req.sampling for req in reqs]
        self.sampling_meta = SamplingMetadata.from_sampling_list(batch_sampling, config, self.batch_size)

        # metrics
        now=time.perf_counter()
        self.batch_metrics = [
            Metrics(arrival_time=req.arrival_time, schedule_time=now, input_token_num=len(req.input_ids)) for req in self.reqs
        ]

        # intermediate states for the batch
        self.mutable_prompt_ids = copy.deepcopy(self.prompt_ids)                    # for prefill in the non-cache scenario, we can modify the requests in place
        self.output_ids: list[list[int]] = [[] for _ in range(self.batch_size)]     # for repetition penality and synchronous generation
        self.finished: list[bool] = [False] * self.batch_size
        self.past_lens: torch.Tensor | None = None
        self.next_tokens: torch.Tensor | None = None
        self.step = 0
        
    def add_sampled_tokens(self, next_tokens: torch.Tensor, past_lens: torch.Tensor, eos_token_id, is_prefill:bool = False) -> None:
        self.next_tokens = next_tokens
        next_tokens_cpu = next_tokens.tolist()
        self.past_lens = past_lens
        self.step += 1

        now = time.perf_counter()
        for batch_idx in range(self.batch_size):
                if self.finished[batch_idx]:
                    continue

                tok = next_tokens_cpu[batch_idx]
                loop = self.reqs[batch_idx].loop
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"BatchRequest, batch_idx: {batch_idx}, before add, len: {len(self.output_ids[batch_idx])}, max: {self.reqs[batch_idx].max_new_tokens}, sampled token={tok}")
                if tok in eos_token_id or len(self.output_ids[batch_idx])+1 >= self.reqs[batch_idx].max_new_tokens:
                    self.finished[batch_idx] = True

                    # self.mutable_prompt_ids, do not append when finshed
                    self.output_ids[batch_idx].append(tok)
                    self.batch_metrics[batch_idx].report(now)

                    if loop is not None:
                        loop.call_soon_threadsafe(self.reqs[batch_idx].token_queue.put_nowait, tok)
                        loop.call_soon_threadsafe(self.reqs[batch_idx].token_queue.put_nowait, None)  # sentinel = stream end
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"BatchRequest: batch_idx: {batch_idx}, finished, len: {len(self.output_ids[batch_idx])}, sampled token={tok}")
                else:
                    self.mutable_prompt_ids[batch_idx].append(tok) if is_prefill else None  # for prefill without kv cache
                    self.output_ids[batch_idx].append(tok)
                    self.batch_metrics[batch_idx].report(now)

                    if loop is not None:
                        loop.call_soon_threadsafe(self.reqs[batch_idx].token_queue.put_nowait, tok)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"BatchRequest: step: {self.step}, finished: {self.finished}")

        self.scheduler.log_stats() if self.scheduler is not None else None

class Scheduler:
    def __init__(self, config: ModelConfig, is_benchmarking: bool):
        self.config = config
        self.max_seqs = config.max_seqs
        self.max_waiting = config.max_waiting

        self.waiting: deque[ModelRequest] = deque()
        self.running_batch: BatchRequest | None = None
        self.metrics_list: list[Metrics] = []

        # for stats
        self.stat_interval = config.stat_interval
        self._last_stat_time = time.perf_counter()  # starting time

        # for benchmark
        self.is_benchmarking = is_benchmarking
        self.req_cnt_since_born:int = 0
        self.total_metrics: list[Metrics] = []
        self.log_name_flag = datetime.now().strftime("%Y%m%d_%H%M%S")   # may be changed in test_benchmark_* functions.

    def add_request(self, req) -> bool:
        if len(self.waiting) >= self.max_waiting:
            return False
        
        self.waiting.append(req)
        return True

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running_batch)

    def schedule(self) -> BatchRequest:            # called by run_loop when idle
        raise NotImplementedError

    def finish_running_batch(self) -> None:
        assert self.running_batch
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"StaticScheduler: finishing batch of size {self.running_batch.batch_size}, batch_metrics: {self.running_batch.batch_metrics}")

        self.metrics_list.extend(self.running_batch.batch_metrics)
        self.log_stats()   # reconcile when metrics_list is empty at `add_sampled_tokens`

        self.req_cnt_since_born += self.running_batch.batch_size

        self.running_batch = None       # reset

    def log_stats(self, is_exitting: bool=False):
        now = time.perf_counter()
        if now - self._last_stat_time < self.stat_interval and not is_exitting:
            return
        self._last_stat_time = now

        if not self.metrics_list:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("empty metrics_list")
            return

        self.total_metrics.extend(self.metrics_list) if self.total_metrics is not None else None

        metrics_list = self.metrics_list
        self.metrics_list = []      # reset

        json_bytes = analyze_stats(metrics_list=metrics_list)
        logger.info(f"req_cnt_since_born: {self.req_cnt_since_born}, stats: {json_bytes.decode()}")
        if self.is_benchmarking:
            log_path = Path(LOG_DIR)
            log_path.mkdir(parents=True, exist_ok=True)
            stats_path = log_path / f'stats.{self.log_name_flag}'
            with open(stats_path, "a+b") as f:
                f.write(json_bytes)
                f.write(b"\n")

class StaticScheduler(Scheduler):
    def __init__(self, config: ModelConfig, is_benchmarking=False):
        super().__init__(config=config, is_benchmarking=is_benchmarking)

    def schedule(self) -> BatchRequest:            # called by run_loop when idle
        if self.running_batch:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"StaticScheduler: re-dispatching batch of size {self.running_batch.batch_size}")
            return self.running_batch

        min_batch_num = min(self.max_seqs, len(self.waiting))
        running = [self.waiting.popleft() for _ in range(min_batch_num)]
        self.running_batch = BatchRequest(running, self.config, scheduler=self)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"StaticScheduler: dispatching batch of size {self.running_batch.batch_size}")
        
        return self.running_batch

