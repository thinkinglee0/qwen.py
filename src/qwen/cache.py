import torch
from collections import deque
import logging
import numpy as np
import time

from qwen.config import ModelConfig


logger = logging.getLogger(__name__)


def cdiv(a: int, b: int) -> int:
    return -(-a//b)

class BlockPool:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free: deque[int] = deque(range(num_blocks))    # logic block index -> physical block id
        self.ref_cnt: list[int] = [0]*num_blocks

    def available(self) -> int:
        return len(self.free)

    def alloc(self) -> int:
        assert len(self.free) > 0
        block_id = self.free.popleft()
        assert self.ref_cnt[block_id] == 0, "alloc a living block"
        self.ref_cnt[block_id] = 1  # occupy
        return block_id

    def incref(self, block_id: int):
        assert self.ref_cnt[block_id] > 0

        self.ref_cnt[block_id] += 1

    def decref(self, block_id: int):
        assert self.ref_cnt[block_id] > 0

        self.ref_cnt[block_id] -= 1
        if self.ref_cnt[block_id] == 0:
            self.free.append(block_id)


class KVCacheData:
    def __init__(self, config: ModelConfig):
        elem = 2 * config.num_hidden_layers * config.num_blocks * config.block_size * config.num_key_value_heads * config.head_dim
        total_bytes = elem * torch.empty((), dtype=config.dtype).element_size()

        # check GPU memory volume
        bytes_per_block = total_bytes // config.num_blocks
        assert config.device is not None
        if config.device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info()
            budget = int(free_bytes * 0.9)          # leave headroom for activations and workspace
            assert total_bytes <= budget, (
                f"kv cache wants {total_bytes / 2**30:.2f} GiB, only {budget / 2**30:.2f} GiB usable; "
                f"set num_blocks <= {budget // bytes_per_block}"
            )

        logger.info(f"start to create kv cache, size: {total_bytes / 1024 / 1024} MB, num_blocks: {config.num_blocks}, block_size: {config.block_size}")

        # ensure k and v per layer adjacent
        kv_caches = [
            torch.zeros(2, config.num_blocks, config.block_size, config.num_key_value_heads, config.head_dim, device=config.device, dtype=config.dtype)
            for _ in range(config.num_hidden_layers)
        ]

        self.k_caches, self.v_caches = [], []
        for kv_cache in kv_caches:
            kv = kv_cache.unbind(0)
            self.k_caches.append(kv[0])
            self.v_caches.append(kv[1])

        logger.info(f"creating kv cache done")
        

class KVCache:
    def __init__(self, config: ModelConfig):
        self.block_size = config.block_size
        self.watermark_blocks = config.num_blocks // 100    # 1% reserved for in-flight Ds

        self.pool = BlockPool(config.num_blocks)
        self.block_tables: dict[str, list[int]] = {}    # request_id -> physical block id list
        self.data = KVCacheData(config)

        # verification
        self.cache_verification_interval = config.cache_verification_interval
        self._last_stat_time = time.perf_counter()  # starting time

    def new_blocks_needed(self, request, num_new_tokens: int):
        total = cdiv(request.num_computed_tokens + num_new_tokens, self.block_size)
        cur = len(self.block_tables.get(request.request_id, []))
        return total-cur

    def verify_invariant(self):
        live = sum(1 for c in self.pool.ref_cnt if c > 0)
        assert len(self.pool.free) + live == self.pool.num_blocks

        # every block with a live refcount must be reachable from some block_table
        owned = {b for tbl in self.block_tables.values() for b in tbl}      # all physical block ids owened by requests
        held = {b for b, c in enumerate(self.pool.ref_cnt) if c > 0}        # all physical block ids with ref_cnt greater than zero, that is, live
        assert held <= owned, f"orphaned blocks (leaked): {sorted(held - owned)}"
        assert owned <= held, f"dangling table entries: {sorted(owned - held)}"

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"cache invariant vierifciation succeeded, interval: {self.cache_verification_interval}")

    def verify_invariant_periodical(self, now: float=time.perf_counter()):
        if now - self._last_stat_time > self.cache_verification_interval:
            self.verify_invariant()
        self._last_stat_time = now

    def allocate_slots(self, request, want: int, respect_watermark: bool = False) -> list[int] | None:
        assert want > 0
        need = self.new_blocks_needed(request, want)
        assert need >= 0

        reserve = self.watermark_blocks if respect_watermark else 0
        if need > self.pool.available() - reserve:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"no new block for {request.request_id}")
            return None

        table = self.block_tables.setdefault(request.request_id, [])
        for _ in range(need):
            table.append(self.pool.alloc())

        # compute-bound
        # start = request.num_computed_tokens
        # slots = [       # physical id
        #     table[pos // self.block_size]*self.block_size + pos % self.block_size   # table {logic block id -> physical block id}
        #     for pos in range(start, start+num_new_tokens)
        # ]

        # optimized by vector
        start = request.num_computed_tokens
        if want == 1:
            slots = [table[start // self.block_size] * self.block_size + start % self.block_size]
        else:
            pos = np.arange(start, start + want, dtype=np.int64)
            tbl = np.asarray(table, dtype=np.int64)
            slots_arr = tbl[pos // self.block_size] * self.block_size + (pos % self.block_size)
            slots = slots_arr.tolist()

        self.verify_invariant_periodical()
        return slots

    def pop_slots(self, request, want: int):
        table = self.block_tables.get(request.request_id)
        assert table

        start = request.num_computed_tokens
        end = start + want
        num_pop = cdiv(end, self.block_size) - cdiv(start, self.block_size)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"start: {start}, end: {end}, num_pop: {num_pop}")
        for _ in range(num_pop):
            phy_block_id = table.pop()
            self.pool.decref(phy_block_id)

        if not table:   # empty
            self.block_tables.pop(request.request_id)

        self.verify_invariant_periodical()

    def free(self, request):
        table = self.block_tables.pop(request.request_id, [])
        for block_id in table:
            self.pool.decref(block_id)

        self.verify_invariant_periodical()

    def get_block_table(self, request) -> list[int] | None:
        return self.block_tables.get(request.request_id)

