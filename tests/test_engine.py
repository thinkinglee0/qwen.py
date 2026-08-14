import pytest
import logging
from pathlib import Path
from datetime import datetime

from constants import *
from qwen.engine import ServingDriver, LLMEngine
from qwen.engine import async_generate, benchmark
from constants import MAX_NEW_TOKEN_NUM, REP_PEN_OFF, TEMP_GREEDY, LOG_DIR
from qwen.metrics import analyze_stats

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
    encoding = request.getfixturevalue(encoding_fixture)
    input_list = request.getfixturevalue(list_fixture)
    B = len(input_list)

    tmp_target_driver.engine.model.config.temperature = TEMP_GREEDY
    tmp_target_driver.engine.model.config.repetition_penalty = REP_PEN_OFF

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
    tmp_target_config.block_size = 16
    tmp_target_config.long_prefill_token_threshold = 32   # force real chunked prefill too

    tmp_target_config.temperature = TEMP_GREEDY
    tmp_target_config.repetition_penalty = REP_PEN_OFF
    tmp_target_config.is_benchmarking = True

    engine = LLMEngine(config=tmp_target_config)
    sch = engine.scheduler

    logger.info("start benchmarking under pressure")
    out_ids, _ = benchmark(engine, batch_for_regular_benchmarking, max_new_tokens=MAX_NEW_TOKEN_NUM)

    out_pressure = []
    for i, o in zip(batch_for_regular_benchmarking, out_ids):
        out_pressure.append(i+o)

    # invariants that only hold if preempt / recompute / re-admission are all correct
    assert len(out_pressure) == req_cnt
    assert sch.completed_req_cnt == req_cnt     # nothing silently dropped
    assert all(len(o) > 0 for o in out_pressure)
    assert not sch.running and not sch.waiting
    sch.cache.verify_invariant()                # every block returned

    # under no pressure
    logger.info("start to call async_generate one by one under no pressure")
    with ServingDriver(engine) as driver:
        for idx, input_ids in enumerate(batch_for_regular_benchmarking):
            out_ids = [tok async for tok in async_generate(driver, input_ids, sampling=None, max_new_tokens=MAX_NEW_TOKEN_NUM)]
            if out_pressure[idx] != out_ids:
                logger.info(f"idx: {idx}, benchmark: |{tokenizer.decode(out_pressure[idx])}|")
                logger.info(f"idx: {idx}, async_gen: |{tokenizer.decode(out_ids)}|")
                assert False

def _test_benchmark(engine: LLMEngine, input_ids:list[list[int]], tok):
    assert engine.scheduler.is_benchmarking

    # clean
    input_ids = [
        i for i in input_ids if len(i) < engine.model.config.max_model_len
    ]

    num_reqs = len(input_ids)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name_flag = f'{engine.model.config.device}.{engine.model.config.max_num_seqs}.{num_reqs}.{time_str}'
    engine.scheduler.log_name_flag = log_name_flag      # pass to `scheduler` for `stats.xxx` log

    logger.info(f"Starting benchmark..., number of requests: {num_reqs}, max_num_seqs: {engine.model.config.max_num_seqs}, max_model_len: {engine.model.config.max_model_len}")
    output_ids, elapsed = benchmark(engine, input_ids, max_new_tokens=1024)
    assert num_reqs == len(output_ids)

    log_path = Path(LOG_DIR)
    log_path.mkdir(parents=True, exist_ok=True)

    # write input+out to file
    output_file = log_path / f'output.{log_name_flag}'
    with open(output_file, "wb") as f:
        for idx, (input, out) in enumerate(zip(input_ids, output_ids)):
            line = f"idx: {idx}, len: {len(input)}-{len(out)}, ||{tok.decode(input)}||\n||{tok.decode(out)}||\n"
            f.write(line.encode())

    # write stats to file
    assert len(engine.scheduler.total_metrics) > 0
    json_bytes = analyze_stats(engine.scheduler.total_metrics)
    logger.info(f"benchmark_stats: {json_bytes.decode()}")
    stats_log_file = log_path / f'benchmark_stats.{log_name_flag}'
    with open(stats_log_file, "wb") as f:
        f.write(json_bytes)

    num_output_ids = sum([len(o) for o in output_ids])
    logger.info(f"Benchmark results: {elapsed} seconds, rate: {num_output_ids/elapsed} r/s")

def test_benchmark_regularly(target_engine_for_regular_benchmarking, batch_for_regular_benchmarking, tokenizer):
    _test_benchmark(target_engine_for_regular_benchmarking, batch_for_regular_benchmarking, tokenizer)

# pytest -x --log-file-level=DEBUG tests/test_engine.py::test_benchmark_sharegpt --max_model_len=256 --req_num=512
# excluded from execution from file, only allowed from specified execution.
@pytest.mark.parametrize("B", [
    1, 
    2, 4, 8, 16, 32, 64,
])
def test_benchmark_sharegpt(target_engine_for_sharegpt_benchmarking, shareGPT_batch_for_sharegpt_benchmarking, tokenizer, B:int):
    target_engine_for_sharegpt_benchmarking.scheduler.max_num_seqs = B

    _test_benchmark(target_engine_for_sharegpt_benchmarking, shareGPT_batch_for_sharegpt_benchmarking, tokenizer)
