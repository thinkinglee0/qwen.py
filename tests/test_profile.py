import pytest
import logging
import time
from pathlib import Path
import torch
from torch.profiler import ProfilerActivity, profile
from torch._C._autograd import DeviceType
from torch._C._profiler import _ExperimentalConfig
import gc
import random
import orjson
from collections import Counter
from datetime import datetime

from constants import *
from qwen.engine import LLMEngine
from qwen.config import ModelConfig
from qwen.scheduler import ModelRequest
from utils import parse_env_list_value

logger = logging.getLogger(__name__)


def parse_chrome_trace(trace_path: Path, profile_steps:int=0):
    with open(trace_path) as f:
        events = orjson.loads(f.read())["traceEvents"]

    if profile_steps > 0:
        ops = Counter(e["name"] for e in events
                      if e.get("ph") == "X" and e.get("cat", "").lower() == "cpu_op")
        # one argmax per decode step -- independent of batch size and cuBLAS tile choice
        assert ops["aten::argmax"] == profile_steps, \
            f"expected {profile_steps} steps, trace has {ops['aten::argmax']}"

    GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
    gpu = [(e["ts"], e["ts"] + e["dur"]) for e in events
        if e.get("ph") == "X" and e.get("cat", "").lower() in GPU_CATS]
    cpu = [(e["ts"], e["ts"] + e["dur"]) for e in events
        if e.get("ph") == "X" and e.get("cat", "").lower() == "cpu_op"]


    def union_duration(spans):
        """Merge overlapping intervals; correct even with multiple CUDA streams."""
        if not spans:
            return 0.0
        
        spans = sorted(spans)
        total = 0.0
        cur_s, cur_e = spans[0]
        for s, e in spans[1:]:
            if s > cur_e:                 # gap -> close current interval
                total += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        return total + (cur_e - cur_s)


    gpu_busy = union_duration(gpu)
    wall = max(e for _, e in cpu) - min(s for s, _ in cpu)
    return gpu_busy, wall

CHROME_TRACE_FILE_NAME = "decode_tracing_chrome.{}.json"
DECODE_STACK_CUDA_FILE_NAME = "decode_stacks_cuda.{}.txt"
DECODE_STACK_CPU_FILE_NAME = "decode_stacks_cpu.{}.txt"

# excluded from execution from file, only allowed from specified execution.
# SWEEP_PROFILE_BATCH_SIZES=512 pytest -x -s tests/test_profile.py::test_profile_decode_idle_fraction
_DEFAULT_PROFILE_BATCH_SIZES = [512]
@pytest.mark.parametrize("batch_size", parse_env_list_value(env_name="SWEEP_PROFILE_BATCH_SIZES", default_value=_DEFAULT_PROFILE_BATCH_SIZES))
@pytest.mark.parametrize("use_d_first_schedule", parse_env_list_value(env_name="SWEEP_PROFILE_USE_D_FIRST_SCHEDULE", default_value=[False]))
def test_profile_decode_idle_fraction(tmp_target_config_for_sharegpt_benchmarking: ModelConfig, batch_size: int, use_d_first_schedule:bool):
    '''
    GPU idle fraction of a *pure decode* step, isolated from prefill.

    Same shape as test_benchmark_sweep_batch_size (fixed input_len,
    ignore EOS) so all `batch_size` requests prefill together and finish
    decode in lockstep -- that gives a steady-state decode window with no
    admissions/preemptions to sample from, instead of the mixed
    prefill+decode average that run_to_completion() would produce.
    '''
    logger.info(f"batch_size: {batch_size}, use_d_first_schedule: {use_d_first_schedule}")

    PROMPT_LEN, WARMUP_STEPS, PROFILE_STEPS = 256, 32, 5
    output_len = WARMUP_STEPS + PROFILE_STEPS*2 + 1    # ensure that it does not hit the max_new_tokens ceiling before profiling finishes.

    kv_drift = PROFILE_STEPS / (PROMPT_LEN + WARMUP_STEPS)
    logger.info(f"KV drift between runs: {kv_drift:.2%}")
    assert kv_drift < 0.05

    cfg = tmp_target_config_for_sharegpt_benchmarking
    cfg.eos_token_id = []      # ignore eos
    cfg.max_num_seqs = batch_size
    cfg.max_model_len = 1024
    cfg.max_num_batched_tokens = 8*1024
    cfg.long_prefill_token_threshold = 8*1024
    cfg.num_blocks = 1024*2
    cfg.max_waiting = batch_size    # exactly one wave, no refill needed
    cfg.use_d_first_schedule = use_d_first_schedule

    assert cfg.device is not None
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name_flag = f'{cfg.max_num_seqs}.{cfg.max_num_batched_tokens}.{cfg.long_prefill_token_threshold}.{cfg.use_d_first_schedule}.{time_str}'


    batch_input_ids = [[random.randrange(cfg.vocab_size) for _ in range(PROMPT_LEN)]
                       for _ in range(batch_size)]

    engine: LLMEngine | None = None
    try:
        engine = LLMEngine(cfg)
        engine.scheduler.log_name_flag = log_name_flag

        for input_ids in batch_input_ids:
            req = ModelRequest(cfg, loop=None, input_ids=input_ids, max_new_tokens=output_len)
            assert engine.scheduler.add_request(req), "request rejected: check num_blocks/max_waiting"

        # warmup: drain the initial prefill wave (CUDA context init, lazy
        # module loading and prefill compute are not representative of
        # steady-state decode) until every running request is decoding.

        # Warm up the compiled function first: inductor's own init allocates CUDA
        # tensors and would trip the sync debug guard.
        torch.cuda.set_sync_debug_mode("default")
        for _ in range(WARMUP_STEPS):
            engine.step()
        assert engine.scheduler.running, "no requests admitted -- check num_blocks / max_num_seqs"
        assert len(engine.scheduler.running) == batch_size, "sequences finished mid-measurement -- raise output_len"
        assert all(r.is_decoding for r in engine.scheduler.running), "still prefilling after warmup_steps -- increase warmup_steps"

        # run A: do not profile, only record wall time
        torch.cuda.set_sync_debug_mode("warn") # or error
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(PROFILE_STEPS):
            engine.step()
        torch.cuda.synchronize()
        wall_clean_us = (time.perf_counter() - t0) * 1e6
        torch.cuda.set_sync_debug_mode("default")

        # run B: interleave
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_stack=True,
                     experimental_config=_ExperimentalConfig(verbose=True, enable_cuda_sync_events=True)) as prof:
            for _ in range(PROFILE_STEPS):
                engine.step()
            torch.cuda.synchronize()

        logger.info("profile finished, start to export stacks")

        # Only device_type==CUDA rows are real kernel executions. CPU-side op
        # rows (aten::mm, flash_attn::...) also carry a nonzero self_cuda/
        # self_device_time_total -- the *same* GPU time attributed to their
        # launched kernel -- so including them here double-counts (this is
        # exactly how torch's own EventList.table() computes its "Self CUDA
        # time total" line, see torch.autograd.profiler_util._build_table).
        ka = prof.key_averages()
        _ATTR = "self_cuda_time_total" if hasattr(ka[0], "self_cuda_time_total") else "self_device_time_total"
        SYNC_ROWS = {"Stream Sync", "Context Sync", "Device Sync", "Event Sync"}
        gpu_device_time_sum_us = sum(
            getattr(e, _ATTR) for e in ka
            if e.device_type == DeviceType.CUDA
            and getattr(e, _ATTR) > 0
            and e.key not in SYNC_ROWS          # sync markers are wait time, not kernel time
        )
        # export_chrome_trace / export_stacks / key_averages() all require the
        # profiler to have actually stopped (i.e. the `with` block exited) --
        # calling them while still inside raises "Profiler didn't finish
        # running" (or, on newer torch, a bare AttributeError on
        # kineto_results being None).
        trace_path = Path(cfg.log_dir) / CHROME_TRACE_FILE_NAME.format(log_name_flag)
        prof.export_chrome_trace(str(trace_path))
        # prof.export_stacks(str(Path(cfg.log_dir) / DECODE_STACK_CUDA_FILE_NAME.format(log_name_flag)), "self_cuda_time_total")
        # prof.export_stacks(str(Path(cfg.log_dir) / DECODE_STACK_CPU_FILE_NAME.format(log_name_flag)), "self_cpu_time_total")

        gpu_busy_from_trace_us, wall_from_trace_us = parse_chrome_trace(trace_path=trace_path, profile_steps=PROFILE_STEPS)

        logger.info(f"batch_size: {batch_size}, profile_steps: {PROFILE_STEPS}, "
                    f"wall_from_trace_us: {wall_from_trace_us:.1f}, wall_clean_us: {wall_clean_us:.1f}, "
                    f"gpu_busy_from_trace_us: {gpu_busy_from_trace_us:.1f}, gpu_device_time_sum_us: {gpu_device_time_sum_us:.1f}")

        assert gpu_device_time_sum_us > 0, \
            "no CUDA device rows in key_averages -- check DeviceType filtering"
        overlap_pct = (gpu_device_time_sum_us - gpu_busy_from_trace_us) / gpu_device_time_sum_us
        logger.info(f"kernel overlap: {overlap_pct:.2%} (near 0 expected on a single stream)")
        assert -0.01 < overlap_pct < 0.5, "sum vs union mismatch -- check trace parsing"

        stack_overhead_pct = (wall_from_trace_us - wall_clean_us) / wall_clean_us
        logger.info(f"with_stack overhead: {stack_overhead_pct:+.1%} "
                    f"(trace wall vs clean wall)")

        gpu_utilization = gpu_busy_from_trace_us / wall_clean_us
        assert 0.0 <= gpu_utilization <= 1.05, f"nonsensical utilization: {gpu_utilization}"

        logger.info(f"for per step, wall_clean_us={wall_clean_us/PROFILE_STEPS:.1f} us/step, "
                    f"gpu_busy_from_trace_us={gpu_busy_from_trace_us/PROFILE_STEPS:.1f} us/step, "
                    f"gpu_idle_fraction={1 - gpu_utilization:.1%}")
        logger.info(f"key_averages len: {len(ka)} distinct rows")
        logger.info(ka.table(sort_by=_ATTR, row_limit=-1))
        logger.info(ka.table(sort_by="self_cpu_time_total", row_limit=-1))
    finally:
        # explicitly release kv cache
        if engine is not None:
            engine.teardown()
            del engine

        gc.collect()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def test_parse_chrome_trace(log_dir: str):
    trace_path = Path(log_dir) / CHROME_TRACE_FILE_NAME
    gpu_busy, wall = parse_chrome_trace(trace_path=trace_path)
    logger.info(f"wall={wall/1e3:.1f}ms  gpu_busy={gpu_busy/1e3:.1f}ms  "
        f"util={100*gpu_busy/wall:.1f}%  idle={(wall-gpu_busy)/1e3:.1f}ms")

