import logging
import torch
import copy
import random
from collections import deque

from qwen.cache import KVCache, KVCacheData, cdiv
from qwen.config import ModelConfig
from qwen.scheduler import SchedulerOutput, ModelRequest, ScheduledInfo, Scheduler

logger = logging.getLogger(__name__)


def verify_slots_continuity(want: int, slots: list[int] | None, req: ModelRequest, cache: KVCache):
    assert slots is not None
    assert want == len(slots) and want == len(set(slots))   # no aliasing within the request

    '''inner-block continuity'''
    block_size = cache.block_size
    start = req.num_computed_tokens
    end = start + want
    table = cache.get_block_table(req)
    assert table is not None
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"want: {want}, block_size: {block_size}, start/end: {start}/{end}, table: {table}, slots: {slots}")

    for i, pos in enumerate(range(start, end)):
        assert slots[i] // block_size == table[pos // block_size]   # logical block -> physical block
        assert slots[i] %  block_size == pos %  block_size          # intra-block offset preserved


def test_cache(tmp_target_config: ModelConfig, seed=0):
    tmp_target_config.num_blocks = 16
    tmp_target_config.block_size = 16
    tmp_target_config.cache_verification_interval = 0.  # always trigger cache invariant verification

    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 100))[0].tolist()    # rectangular tensor
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids, sampling=None, max_new_tokens=1000)
    req1.request_id = "req1"

    cache = KVCache(tmp_target_config)
    cache.verify_invariant_periodical()

    # make free deque of block pool out-of-order
    free_list = list(cache.pool.free)
    random.Random(seed).shuffle(free_list)
    cache.pool.free = deque(free_list)

    # step 1: alloc 20 tokens
    want = 20
    slots = cache.allocate_slots(req1, want)
    table1 = cache.get_block_table(req1)
    assert table1 is not None and len(table1) == 2                      # cdiv(want, cache.block_size) ceil(100/16)
    assert table1 != list(range(table1[0], table1[0] + len(table1)))    # verify increment
    verify_slots_continuity(want, slots, req1, cache)

    table = cache.block_tables[req1.request_id]
    assert len(table) == cdiv(want, cache.block_size)

    # step 2: advance
    assert req1.num_computed_tokens == 0
    req1.num_computed_tokens += want

    # step 3-1: alloc 8, total=20+8, less than 32, two times of block_size
    cache_snapshot = copy.deepcopy(cache)   # deep copy
    want = 8
    slots = cache_snapshot.allocate_slots(req1, want)
    verify_slots_continuity(want, slots, req1, cache_snapshot)
    table = cache_snapshot.block_tables[req1.request_id]
    assert cdiv(want+req1.num_computed_tokens, cache_snapshot.block_size) == 2
    assert len(table) == 2

    # step 3-2: alloc 12, total=20+12, divisible without remainder to block_size
    cache_snapshot = copy.deepcopy(cache)   # deep copy
    want = 12
    slots = cache_snapshot.allocate_slots(req1, want)
    verify_slots_continuity(want, slots, req1, cache_snapshot)
    table = cache_snapshot.block_tables[req1.request_id]
    assert cdiv(want+req1.num_computed_tokens, cache_snapshot.block_size) == 2
    assert len(table) == 2

    # step 3-3: alloc 13, total=20+13, divisible with remainder to block_size
    cache_snapshot = copy.deepcopy(cache)   # deep copy
    want = 13
    slots = cache_snapshot.allocate_slots(req1, want)
    verify_slots_continuity(want, slots, req1, cache_snapshot)
    table = cache_snapshot.block_tables[req1.request_id]
    assert cdiv(want+req1.num_computed_tokens, cache_snapshot.block_size) == 3
    assert len(table) == 3

    # step 4: pop slots just allocated
    want = 13
    slots = cache.allocate_slots(req1, want)
    verify_slots_continuity(want, slots, req1, cache)
    cache.pop_slots(req1, want)
    table = cache.block_tables[req1.request_id]
    assert len(table) == cdiv(req1.num_computed_tokens, cache.block_size)

    # step 5: pop all
    want = req1.num_computed_tokens
    req1.num_computed_tokens = 0    # reset
    cache.pop_slots(req1, want)
    table = cache.block_tables.get(req1.request_id, None)
    assert table is None

def test_pool_exhausted(tmp_target_config: ModelConfig):
    tmp_target_config.num_blocks = 1
    tmp_target_config.block_size = 16
    tmp_target_config.cache_verification_interval = 0.  # always trigger cache invariant verification

    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 100))[0].tolist()    # rectangular tensor
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids, sampling=None, max_new_tokens=1000)
    req1.request_id = "req1"

    cache = KVCache(tmp_target_config)

    want = len(req1.input_ids)
    slots = cache.allocate_slots(req1, want)
    assert slots is None and cache.block_tables == {} and cache.pool.available() == tmp_target_config.num_blocks

def test_free(tmp_target_config: ModelConfig, seed=0):
    tmp_target_config.num_blocks = 10
    tmp_target_config.block_size = 16
    tmp_target_config.cache_verification_interval = 0.  # always trigger cache invariant verification

    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 100))[0].tolist()    # rectangular tensor
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids, sampling=None, max_new_tokens=1000)
    req1.request_id = "req1"

    cache = KVCache(tmp_target_config)

    # make free deque of block pool out-of-order
    free_list = list(cache.pool.free)
    random.Random(seed).shuffle(free_list)
    cache.pool.free = deque(free_list)

    want = len(req1.input_ids)
    slots = cache.allocate_slots(req1, want)
    table1 = cache.get_block_table(req1)
    assert table1 is not None and len(table1) == 7                      # cdiv(want, cache.block_size) ceil(100/16)
    assert table1 != list(range(table1[0], table1[0] + len(table1)))    # verify increment
    verify_slots_continuity(want, slots, req1, cache)

    # free
    cache.free(req1)

    # verify
    assert cache.get_block_table(req1) is None
    assert cache.pool.available() == cache.pool.num_blocks

def test_watermark(tmp_target_config: ModelConfig):
    tmp_target_config.num_blocks = 100
    tmp_target_config.block_size = 10
    tmp_target_config.cache_verification_interval = 0.  # always trigger cache invariant verification

    input_ids = torch.randint(0, tmp_target_config.vocab_size, (1, 2000))[0].tolist()    # rectangular tensor
    req1 = ModelRequest(tmp_target_config, loop=None, input_ids=input_ids, sampling=None, max_new_tokens=1000)
    req1.request_id = "req1"

    cache = KVCache(tmp_target_config)
    assert cache.watermark_blocks == 1

    assert cache.allocate_slots(req1, 990) is not None
    cache.free(req1)

    assert cache.allocate_slots(req1, 991) is not None
    cache.free(req1)

    assert cache.allocate_slots(req1, 990, respect_watermark=True) is not None
    cache.free(req1)

    assert cache.allocate_slots(req1, 991, respect_watermark=True) is None
    cache.free(req1)

    assert cache.allocate_slots(req1, 1000) is not None
    cache.free(req1)

    assert cache.allocate_slots(req1, 1001) is None
    cache.free(req1)

    assert cache.allocate_slots(req1, 1001, respect_watermark=True) is None
    cache.free(req1)



