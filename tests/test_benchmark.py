import pytest
import logging
import time
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import orjson
from typing import Any
import torch
import gc
import random

from constants import *
from qwen.engine import LLMEngine
from qwen.engine import benchmark
from qwen.metrics import analyze_metrics
from qwen.utils import sample_sharegpt
from qwen.config import ModelConfig
from qwen.constants import DEFAULT_MAX_NEW_TOKEN
from utils import parse_env_list_value

logger = logging.getLogger(__name__)


def _test_benchmark(engine: LLMEngine, input_ids:list[list[int]], tok, max_new_tokens=DEFAULT_MAX_NEW_TOKEN):
    assert engine.scheduler.is_benchmarking
    cfg = engine.model.config
    assert cfg.device is not None

    # clean
    input_ids = [
        i for i in input_ids if len(i) < engine.model.config.max_model_len
    ]

    num_reqs = len(input_ids)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name_flag = f'{cfg.device.type}.{cfg.max_num_seqs}.{num_reqs}.{cfg.max_num_batched_tokens}.{cfg.long_prefill_token_threshold}.{cfg.num_blocks}.{time_str}'
    engine.scheduler.log_name_flag = log_name_flag      # pass to `scheduler` for `metrics.xxx` log

    logger.info(f"Starting benchmark..., number of requests: {num_reqs}, max_num_seqs: {engine.model.config.max_num_seqs}, max_model_len: {engine.model.config.max_model_len}")
    output_ids, elapsed = benchmark(engine, input_ids, max_new_tokens=max_new_tokens)
    assert num_reqs == len(output_ids)

    log_path = Path(engine.model.config.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # write input+out to file
    output_file = log_path / f'output.{log_name_flag}'
    with open(output_file, "wb") as f:
        for idx, (input, out) in enumerate(zip(input_ids, output_ids)):
            line = f"idx: {idx}, len: {len(input)}-{len(out)}, ||{tok.decode(input)}||\n||{tok.decode(out)}||\n"
            f.write(line.encode())

    # write metrics to file
    assert len(engine.scheduler.total_metrics) > 0
    json_bytes = analyze_metrics(req_metrics_list=engine.scheduler.total_metrics,
                                 sch_metrics=engine.scheduler.sch_metrics,
                                 is_benchmarking=True, config=engine.model.config)
    logger.info(f"benchmark_metrics: {json_bytes.decode()}")
    stats_log_file = log_path / f'benchmark_metrics.{log_name_flag}.json'
    with open(stats_log_file, "wb") as f:
        f.write(json_bytes)
        f.write(b"\n")
        f.flush()

    num_output_ids = sum([len(o) for o in output_ids])
    logger.info(f"Benchmark results: {elapsed} seconds, rate: {num_output_ids/elapsed} /s")

def test_benchmark_on_pc(target_engine_for_pc_benchmarking, batch_for_regular_benchmarking, tokenizer):
    _test_benchmark(target_engine_for_pc_benchmarking, batch_for_regular_benchmarking, tok=tokenizer)

# pytest -x --log-file-level=DEBUG tests/test_benchmark.py::test_benchmark_sharegpt --max_model_len=512 --req_num=512 --max_num_seqs=16
# excluded from execution from file, only allowed from specified execution.
def test_benchmark_sharegpt(target_engine_for_sharegpt_benchmarking, sharegpt_batch, tokenizer):
    _test_benchmark(target_engine_for_sharegpt_benchmarking, sharegpt_batch, tok=tokenizer)

_DEFAULT_BATCH_SIZES = [
    512,      # warm up
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
_DEFAULT_BLOCKS = [1024*6]  # 18G
# excluded from execution from file, only allowed from specified execution.
# SWEEP_BATCH_SIZES=64,128 pytest -x tests/test_benchmark.py::test_benchmark_sweep_batch_size
@pytest.mark.parametrize("batch_size", parse_env_list_value(env_name="SWEEP_BATCH_SIZES", default_value=_DEFAULT_BATCH_SIZES))
@pytest.mark.parametrize("num_blocks", parse_env_list_value(env_name="SWEEP_BLOCKS", default_value=_DEFAULT_BLOCKS))
def test_benchmark_sweep_batch_size(tmp_target_config_for_sharegpt_benchmarking: ModelConfig, tokenizer, batch_size:int, num_blocks:int, log_dir):
    '''
    1. fixed input len 512, fixed output len 128
    2. ignore EOS
    '''
    cfg = tmp_target_config_for_sharegpt_benchmarking
    cfg.eos_token_id = []      # ignore eos
    cfg.max_num_seqs = batch_size
    cfg.max_model_len = 1024
    cfg.max_num_batched_tokens = 8*1024
    cfg.long_prefill_token_threshold = 8*1024
    cfg.num_blocks = num_blocks
    req_num = max(64, 10*batch_size)
    cfg.max_waiting = req_num
    cfg.log_dir = log_dir   # log_dir = str(Path(log_dir) / "batch_size")

    # fixed-length input
    input_len, output_len = 512, 128
    batch_input_ids = [[random.randrange(cfg.vocab_size) for _ in range(input_len)]
                   for _ in range(req_num)]

    try:
        engine = LLMEngine(cfg)
        _test_benchmark(engine, batch_input_ids, tok=tokenizer, max_new_tokens=output_len)
    finally:
        # explicitly release kv cache
        engine.teardown()
        del engine

        gc.collect()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# excluded from execution from file, only allowed from specified execution.
# SWEEP_BATCHED_TOKENS=64,128 pytest -x tests/test_benchmark.py::test_benchmark_sweep_batched_tokens_and_long_prefill_token_threshold
_DEFAULT_BATCHED_TOKENS = [1024, 2048, 4096, 8192]
_DEFAULT_PREFILL_SEQS = [1, 2, 4, 8]
@pytest.mark.parametrize("batch_size", [320])
@pytest.mark.parametrize("max_num_batched_tokens",
                         parse_env_list_value(env_name="SWEEP_BATCHED_TOKENS", default_value=_DEFAULT_BATCHED_TOKENS))
@pytest.mark.parametrize("num_prefill_seqs", 
                         parse_env_list_value(env_name="SWEEP_PREFILL_SEQS", default_value=_DEFAULT_PREFILL_SEQS))
def test_benchmark_sweep_batched_tokens_and_long_prefill_token_threshold(tmp_target_config_for_sharegpt_benchmarking, tokenizer, batch_size:int, log_dir,
                          max_num_batched_tokens:int, 
                          num_prefill_seqs:int,):
    cfg = tmp_target_config_for_sharegpt_benchmarking

    cfg.max_num_seqs = batch_size
    cfg.max_model_len = 1024
    cfg.max_num_batched_tokens = max_num_batched_tokens
    assert max_num_batched_tokens >= 2 * batch_size, "budget must be non-binding for decode"
    cfg.long_prefill_token_threshold = (max_num_batched_tokens - batch_size) // num_prefill_seqs
    assert cfg.long_prefill_token_threshold >= 128, f"threshold {cfg.long_prefill_token_threshold} below the free-chunk floor"
    cfg.num_blocks = 6*1024     # 18G
    req_num = max(64, 10*batch_size)
    cfg.max_waiting = req_num
    cfg.log_dir = log_dir   # log_dir = str(Path(log_dir) / "max_num_seqs")

    sharegpt_batch = sample_sharegpt(SHARE_GPT_FILE_NAME, tokenizer,  num_requests=req_num, 
                                     max_p_len=cfg.max_model_len//2, max_model_len=cfg.max_model_len)

    try:
        engine = LLMEngine(cfg)
        _test_benchmark(engine, sharegpt_batch, tok=tokenizer, max_new_tokens=cfg.max_model_len)
    finally:
        # explicitly release kv cache
        engine.teardown()
        del engine

        gc.collect()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

# pytest tests/test_benchmark.py::test_parse_metrics_sweep_batch_size --log_dir=log_vast/log3
def test_parse_metrics_sweep_batch_size(log_dir: str):
    dir_path = Path(log_dir)
    assert dir_path.exists()

    benchmark_files = [file for file in dir_path.glob("benchmark_metrics.*") if file.is_file()]
    # benchmark_files = [
    #     dir_path / "benchmark_metrics.cuda.256.2560.8192.8192.6144.20260901_063347.json",
    #     dir_path / "benchmark_metrics.cuda.320.3200.8192.8192.6144.20260901_063512.json",
    #     dir_path / "benchmark_metrics.cuda.384.3840.8192.8192.6144.20260901_063651.json",
    #     dir_path / "benchmark_metrics.cuda.448.4480.8192.8192.6144.20260901_063841.json",
    #     dir_path / "benchmark_metrics.cuda.512.5120.8192.8192.6144.20260901_064045.json",
    # ]
    '''
    fileds of a file name
    1   device type
    2   max_num_seqs, batch size
    3   num_seqs
    4   max_num_batched_tokens
    5   long_prefill_token_threshold
    6   num_blocks
    '''
    benchmarks = {}
    for file in benchmark_files:
        fields = file.name.split(".")
        logger.info(f"len: {len(fields)}, {fields}")
        if len(fields) != 9:
            continue
        batch = int(fields[2])

        with open(file) as f:
            raw: dict[str, Any] = orjson.loads(f.read())
            benchmarks[batch] = raw

    sorted_batchs = sorted(benchmarks)
    benchmarks = sorted(benchmarks.items())

    prefill, tpot, itl, throughput = [], [], [], []
    tpop_p90, itl_p90 = [], []
    for _, b in benchmarks:
        prefill.append(float(b["prefill"]["mean"]))
        tpot.append(float(b["tpot"]["mean"]))
        itl.append(float(b["itls"]["mean"]))
        tpop_p90.append(float(b["tpot"]["mean"]))
        itl_p90.append(float(b["itls"]["mean"]))
        throughput.append(float(b["basic"]["tok_throughput"]))

    prefill_ratio, tpot_ratio, itl_ratio, throughput_ratio = [], [], [], []
    for idx in range(1, len(sorted_batchs)):
        prefill_ratio.append(round(prefill[idx] / prefill[idx-1], 2))
        tpot_ratio.append(round(tpot[idx] / tpot[idx-1], 2))
        itl_ratio.append(round(itl[idx] / itl[idx-1], 2))
        throughput_ratio.append(round(throughput[idx] / throughput[idx-1], 2))

    logger.info(f"batch: {sorted_batchs}")
    logger.info(f"ttft: {prefill}")
    logger.info(f"ttft_ratio: {prefill_ratio}")
    logger.info(f"tpot: {tpot}")
    logger.info(f"tpot_ratio: {tpot_ratio}")
    logger.info(f"itl: {itl}")
    logger.info(f"itl_ratio: {itl_ratio}")
    logger.info(f"throughput: {throughput}")
    logger.info(f"throughput_ratio: {throughput_ratio}")

    for idx, (gain, cost) in enumerate(zip(throughput_ratio, tpot_ratio)):
        if cost > gain:
            logger.info(f"last profitable doubling ends at batch: {sorted_batchs[idx]} -> {sorted_batchs[idx+1]} (gain/cost > 1)")
            break
