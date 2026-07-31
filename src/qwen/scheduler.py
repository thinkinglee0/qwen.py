from collections import deque
import logging
import threading

from qwen.request import ModelRequest

logger = logging.getLogger(__name__)

class StaticScheduler:
    def __init__(self, max_seqs: int, max_waiting: int):
        self.max_seqs = max_seqs
        self.max_waiting = max_waiting
        self.waiting: deque[ModelRequest] = deque()
        self.running: list[ModelRequest] = []
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

    def add_request(self, req) -> bool:
        with self.lock:
            if len(self.waiting) >= self.max_waiting:
                return False
            
            self.waiting.append(req)
            self.cond.notify()          # wake run_loop if it's waiting
            return True

    def wait_for_work(self):            # called by run_loop when idle
        with self.lock:
            while not self.waiting:
                logger.info("Scheduler: waiting for work")
                self.cond.wait()        # releases lock, blocks until notify

            self.running = []   # empty the running list

            min_batch_num = min(self.max_seqs, len(self.waiting))
            batch = [self.waiting.popleft() for _ in range(min_batch_num)]
            self.running = batch
            logger.info(f"Scheduler: dispatching batch of size {len(batch)}")
            return batch


