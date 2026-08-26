import pytest
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import orjson
from typing import Any
import math
import torch
import gc

from constants import *
from qwen.engine import ServingDriver, LLMEngine
from qwen.engine import async_generate, benchmark
from constants import MAX_NEW_TOKEN_NUM, LOG_DIR
from qwen.metrics import analyze_metrics
from qwen.utils import round_floats
from qwen.utils import sample_sharegpt

logger = logging.getLogger(__name__)

# target model == reference model?
@pytest.mark.parametrize(
    ("encoding_fixture", "list_fixture"),
    [
        pytest.param("solo_encoding", "solo_input_ids_list", id="single"),
        pytest.param("batch_encoding", "batch_input_ids_list", id="batch"),
    ],
)
@pytest.mark.asyncio
async def test_generation_compared_with_reference(tmp_target_driver: ServingDriver, ref_model, tokenizer, request, encoding_fixture, list_fixture):
    encoding = request.getfixturevalue(encoding_fixture).to(ref_model.device)
    input_list = request.getfixturevalue(list_fixture)
    B = len(input_list)

    tmp_target_driver.engine.model.config.do_sample = False
    tmp_target_driver.engine.model.config.do_penalities = False

    # target model
    target_output_token_ids = [[] for _ in range(B)]
    for batch_idx in range(B):
        tokens = [tok async for tok in async_generate(tmp_target_driver, input_list[batch_idx], sampling=None, max_new_tokens=MAX_NEW_TOKEN_NUM)]
        target_output_token_ids[batch_idx] = tokens

    # reference
    original_repetition_penalty = ref_model.generation_config.repetition_penalty
    try:
        ref_model.generation_config.repetition_penalty = REP_PEN_OFF
        ref_output = ref_model.generate(        # tensor output, shape [B, T]
            **encoding,
            max_new_tokens=MAX_NEW_TOKEN_NUM,
            do_sample=False, temperature=None,
            top_p=None, top_k=None, num_beams=1,
        )
    finally:
        ref_model.generation_config.repetition_penalty = original_repetition_penalty

    # compare with reference model
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"attention_mask: {encoding.attention_mask}")
    padding_count = (encoding.attention_mask == 0).sum(dim=1).tolist()      # [B]
    for batch_idx in range(B):
        ref_output_text = tokenizer.decode(ref_output[batch_idx, padding_count[batch_idx]:])
        target_output_text = tokenizer.decode(target_output_token_ids[batch_idx])
        logger.info(f"target_output_text[{batch_idx}]: |{target_output_text}|, len: {len(target_output_token_ids[batch_idx])}")
        logger.info(f"   ref_output_text[{batch_idx}]: |{ref_output_text}|, len: {len(ref_output[batch_idx, padding_count[batch_idx]:])}")
        if target_output_text != ref_output_text:
            logger.info(f"comparison result for batch_idx={batch_idx}: differ")
        else:
            logger.info(f"comparison result for batch_idx={batch_idx}: match")
        
        assert target_output_text == ref_output_text

@pytest.mark.parametrize(
    ("list_fixture"),
    [
        pytest.param("solo_input_ids_list", id="single"),
        pytest.param("batch_input_ids_list", id="batch"),
    ],
)
@pytest.mark.asyncio
async def test_generations_differentiation(tmp_target_driver: ServingDriver, tokenizer, list_fixture, request):
    input_list: list[list[int]] = request.getfixturevalue(list_fixture)

    logger.info(f"start the first turn")
    ids0 = []
    for input_ids in input_list:
        tokens = [tok async for tok in async_generate(tmp_target_driver, input_ids, sampling=None, max_new_tokens=MAX_NEW_TOKEN_NUM)]
        ids0.append(tokens)

    logger.info(f"start the second turn")
    ids1 = []
    for input_ids in input_list:
        tokens = [tok async for tok in async_generate(tmp_target_driver, input_ids, sampling=None, max_new_tokens=MAX_NEW_TOKEN_NUM)]
        ids1.append(tokens)

    for batch_idx in range(len(input_list)):
        logger.info(f"batch_idx: {batch_idx}, output-0: |{tokenizer.decode(ids0[batch_idx])}|, len: {len(ids0[batch_idx])}")
        logger.info(f"batch_idx: {batch_idx}, output-1: |{tokenizer.decode(ids1[batch_idx])}|, len: {len(ids1[batch_idx])}")

    assert ids0 != ids1

@pytest.mark.asyncio
async def test_generation_under_kv_pressure(tmp_target_config, batch_for_regular_benchmarking, tokenizer):
    req_cnt = len(batch_for_regular_benchmarking)
    logger.info(f"request count: {req_cnt}")

    # Sized so the pool cannot hold all concurrent requests -> preemption + recompute is forced.
    tmp_target_config.num_blocks = 8                      # ~128 tokens total: guarantees eviction
    tmp_target_config.block_size = 256
    tmp_target_config.long_prefill_token_threshold = 32   # force real chunked prefill too

    tmp_target_config.do_sample = False
    tmp_target_config.do_penalities = False
    tmp_target_config.is_benchmarking = True

    engine = LLMEngine(config=tmp_target_config)
    sch = engine.scheduler

    logger.info("start benchmarking under pressure")
    out_no_pressure, _ = benchmark(engine, batch_for_regular_benchmarking, max_new_tokens=MAX_NEW_TOKEN_NUM)

    out_pressure = []
    for i, o in zip(batch_for_regular_benchmarking, out_no_pressure):
        out_pressure.append(i+o)

    # invariants that only hold if preempt / recompute / re-admission are all correct
    assert len(out_pressure) == req_cnt
    assert sch.sch_metrics.num_finished == req_cnt     # nothing silently dropped
    assert all(len(o) > 0 for o in out_pressure)
    assert not sch.running and not sch.waiting
    sch.cache.verify_invariant()                # every block returned

    # under no pressure
    logger.info("start to call async_generate one by one under no pressure")
    mismatch_cnt = 0
    with ServingDriver(engine) as driver:
        for idx, input_ids in enumerate(batch_for_regular_benchmarking):
            out_no_pressure = [tok async for tok in async_generate(driver, input_ids, sampling=None, max_new_tokens=MAX_NEW_TOKEN_NUM)]
            if out_pressure[idx] != out_no_pressure:
                len_p, len_n = len(out_pressure[idx]), len(out_no_pressure)
                logger.info(f"idx: {idx}, len: {len_p}, benchmark: |{tokenizer.decode(out_pressure[idx])}|")
                logger.info(f"idx: {idx}, len: {len_n}, async_gen: |{tokenizer.decode(out_no_pressure)}|")
                if tmp_target_config.device.type == "cpu":
                    assert False
                else:
                    mismatch_cnt += 1
                    logger.warning(f"idx: {idx}, outputs under pressure and no pressure mismatch")

    logger.info(f"mismatch / total: {mismatch_cnt} / {req_cnt}")

def _test_benchmark(engine: LLMEngine, input_ids:list[list[int]], tok, max_new_tokens=1024):
    assert engine.scheduler.is_benchmarking
    assert engine.model.config.device is not None

    # clean
    input_ids = [
        i for i in input_ids if len(i) < engine.model.config.max_model_len
    ]

    num_reqs = len(input_ids)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name_flag = f'{engine.model.config.device.type}.{engine.model.config.max_num_seqs}.{num_reqs}.{time_str}'
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

# pytest -x --log-file-level=DEBUG tests/test_engine.py::test_benchmark_sharegpt --max_model_len=512 --req_num=512 --max_num_seqs=16
# excluded from execution from file, only allowed from specified execution.
def test_benchmark_sharegpt(target_engine_for_sharegpt_benchmarking, sharegpt_batch, tokenizer):
    _test_benchmark(target_engine_for_sharegpt_benchmarking, sharegpt_batch, tok=tokenizer)

# excluded from execution from file, only allowed from specified execution.
_DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
import os
def _get_batch_sizes() -> list[int]:
    env = os.environ.get("SWEEP_BATCH_SIZES")
    if env is None:
        return _DEFAULT_BATCH_SIZES
    return [int(b) for b in env.split(",")]

# SWEEP_BATCH_SIZES=64,128 pytest -x tests/test_engine.py::test_benchmark_sharegpt_sweep_batch_size
@pytest.mark.parametrize("B", _get_batch_sizes())
def test_benchmark_sharegpt_sweep_batch_size(tmp_target_config_for_sharegpt_benchmarking, tokenizer, B:int, log_dir):
    cfg = tmp_target_config_for_sharegpt_benchmarking

    cfg.max_num_seqs = B
    cfg.max_model_len = 2*1024
    cfg.max_num_batched_tokens = 2*1024
    cfg.long_prefill_token_threshold = 1024//4
    cfg.num_blocks = 6*1024     # 18G
    req_num = max(64, 10*B)
    cfg.max_waiting = req_num
    cfg.log_dir = log_dir   # log_dir = str(Path(log_dir) / "batch_size")

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

def test_parse_metrics(log_dir: str):
    dir_path = Path(log_dir)
    assert dir_path.exists()

    benchmark_files = [file for file in dir_path.glob("benchmark_metrics.*") if file.is_file()]

    # alternative construction
    # benchmarks: list[dict[str, Any]] = [{} for _ in range(len(benchmark_files))]
    benchmarks = [{}] * len(benchmark_files)
    for file in benchmark_files:
        fields = file.name.split(".")
        assert len(fields) > 3
        batch = int(fields[2])
        batch_idx = int(math.log2(batch))
        assert 1 << batch_idx == batch

        with open(file) as f:
            raw: dict[str, Any] = orjson.loads(f.read())
            benchmarks[batch_idx] = raw

    prefill, tpot, itls, throughput = [], [], [], []
    for b in benchmarks:
        assert b
        prefill.append(float(b["prefill"]["mean"]))
        tpot.append(float(b["tpot"]["mean"]))
        itls.append(float(b["itls"]["mean"]))
        throughput.append(float(b["basic"]["tok_throughput"]))

    prefill_rate, tpot_rate, itls_rate, throughput_rate = [], [], [], []
    for idx in range(1, len(prefill)):
        prefill_rate.append(round(prefill[idx] / prefill[idx-1], 2))
        tpot_rate.append(round(tpot[idx] / tpot[idx-1], 2))
        itls_rate.append(round(itls[idx] / itls[idx-1], 2))
        throughput_rate.append(round(throughput[idx] / throughput[idx-1], 2))

    logger.info(f"batch  : {[1<<i for i in range(len(benchmark_files))]}")
    logger.info(f"prefill: {prefill}")
    logger.info(f"rate   : {prefill_rate}")
    logger.info(f"tpot   : {tpot}")
    logger.info(f"rate   : {tpot_rate}")
    logger.info(f"itls   : {itls}")
    logger.info(f"rate   : {itls_rate}")
    logger.info(f"throughput   : {throughput}")
    logger.info(f"rate   : {throughput_rate}")

