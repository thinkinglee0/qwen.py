from collections import deque
import logging
import copy
import torch

from qwen.request import ModelRequest
from qwen.sampling import SamplingMetadata
from qwen.config import ModelConfig

logger = logging.getLogger(__name__)


class BatchRequest:
    def __init__(self, reqs: list[ModelRequest], config, scheduler = None):
        self.scheduler = scheduler
        self.reqs = reqs
        self.batch_size = len(reqs)

        self.prompt_ids = [req.input_ids for req in reqs]
        batch_sampling = [req.sampling for req in reqs]
        self.sampling_meta = SamplingMetadata.from_sampling_list(batch_sampling, config, self.batch_size)

        # intermediate states for the batch
        self.mutable_prompt_ids = copy.deepcopy(self.prompt_ids)      # for prefill in the non-cache scenario, we can modify the requests in place
        self.output_ids: list[list[int]] = [[] for _ in range(self.batch_size)]
        self.finished: list[bool] = [False] * self.batch_size
        self.past_lens: torch.Tensor | None = None
        self.next_tokens: torch.Tensor | None = None

    def add_sampled_tokens(self, next_tokens: torch.Tensor, past_lens: torch.Tensor, eos_token_id, is_prefill:bool = False) -> None:
        self.next_tokens = next_tokens
        next_tokens_cpu = next_tokens.tolist()
        self.past_lens = past_lens

        for batch_idx in range(self.batch_size):
                if self.finished[batch_idx]:
                    continue

                tok = next_tokens_cpu[batch_idx]
                self.output_ids[batch_idx].append(tok)
                # if logger.isEnabledFor(logging.DEBUG):
                #     logger.debug(f"BatchRequest: batch_idx={batch_idx}, len: {len(self.output_ids[batch_idx])}, sampled token={tok}")
                self.mutable_prompt_ids[batch_idx].append(tok) if is_prefill else None

                loop = self.reqs[batch_idx].loop
                if tok in eos_token_id or len(self.output_ids[batch_idx]) >= self.reqs[batch_idx].max_new_tokens:
                    self.finished[batch_idx] = True
                    if loop is not None:
                        loop.call_soon_threadsafe(self.reqs[batch_idx].token_queue.put_nowait, tok)
                        loop.call_soon_threadsafe(self.reqs[batch_idx].token_queue.put_nowait, None)  # sentinel = stream end
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"BatchRequest: batch_idx={batch_idx}, finished")
                else:
                    if loop is not None:
                        loop.call_soon_threadsafe(self.reqs[batch_idx].token_queue.put_nowait, tok)

        if all(self.finished) and self.scheduler is not None:
            self.scheduler.finish_cur_batch()

class Scheduler:
    def add_request(self, req) -> bool:
        raise NotImplementedError

    def has_unfinished(self) -> bool:
        raise NotImplementedError

    def scheduler(self) -> BatchRequest:            # called by run_loop when idle
        raise NotImplementedError

    def finish_cur_batch(self) -> None:
        raise NotImplementedError

class StaticScheduler(Scheduler):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.max_seqs = config.max_seqs
        self.max_waiting = config.max_waiting
        self.waiting: deque[ModelRequest] = deque()
        self.running: list[ModelRequest] = []

    def add_request(self, req) -> bool:
        if len(self.waiting) >= self.max_waiting:
            return False
        
        self.waiting.append(req)
        return True

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def scheduler(self) -> BatchRequest:            # called by run_loop when idle
        if self.running:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"StaticScheduler: re-dispatching batch of size {len(self.running)}")
            return BatchRequest(self.running, self.config, self)     # static: no admission mid-batch

        min_batch_num = min(self.max_seqs, len(self.waiting))
        self.running = [self.waiting.popleft() for _ in range(min_batch_num)]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"StaticScheduler: dispatching batch of size {len(self.running)}")
        return BatchRequest(self.running, self.config, self)

    def finish_cur_batch(self) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"StaticScheduler: finishing batch of size {len(self.running)}")
        self.running = []   #empty the running list after updating the requests


