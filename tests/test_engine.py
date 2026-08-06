import pytest
import logging
from pathlib import Path
from datetime import datetime

from constants import *
from qwen.engine import async_generate, generate, benchmark
from constants import MAX_NEW_TOKEN_NUM, REP_PEN_OFF, TEMP_GREEDY, LOG_DIR
from qwen.engine import ServingDriver, LLMEngine
from qwen.metrics import analyze_stats
from qwen.model import QwenForCausalLM
from qwen.scheduler import StaticScheduler
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
def test_generation_compared_with_reference(target_model_with_function_scope, ref_model, tokenizer, request, encoding_fixture, list_fixture):
    encoding = request.getfixturevalue(encoding_fixture)
    input_list = request.getfixturevalue(list_fixture)
    B = len(input_list)

    target_model_with_function_scope.config.repetition_penalty = REP_PEN_OFF
    target_model_with_function_scope.config.temperature = TEMP_GREEDY
    target_output_token_ids = generate(target_model_with_function_scope, input_list, max_new_tokens=MAX_NEW_TOKEN_NUM)

    try:
        original_repetition_penalty = ref_model.generation_config.repetition_penalty
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
def test_generations_differentiation(target_model, tokenizer, list_fixture, request):
    input_list = request.getfixturevalue(list_fixture)

    sync_ids0 = generate(target_model, input_list, max_new_tokens=MAX_NEW_TOKEN_NUM)
    sync_ids1 = generate(target_model, input_list, max_new_tokens=MAX_NEW_TOKEN_NUM)

    for batch_idx in range(len(input_list)):
        logger.info(f"batch_idx: {batch_idx}, output-0: |{tokenizer.decode(sync_ids0[batch_idx])}|, len: {len(sync_ids0[batch_idx])}")
        logger.info(f"batch_idx: {batch_idx}, output-1: |{tokenizer.decode(sync_ids1[batch_idx])}|, len: {len(sync_ids1[batch_idx])}")

    assert sync_ids0 != sync_ids1

# synchronous == asynchronous for target model?
@pytest.mark.parametrize(
    ("list_fixture"),
    [
        pytest.param("solo_input_ids_list", id="single"),
        pytest.param("batch_input_ids_list", id="batch"),
    ],
)
@pytest.mark.asyncio
async def test_streaming_generation(target_driver_with_function_scope: ServingDriver, tokenizer, request, list_fixture):
    input_list = request.getfixturevalue(list_fixture)
    B = len(input_list)

    TEMP_GREEDY = 0.
    target_driver_with_function_scope.engine.model.config.temperature = TEMP_GREEDY

    # async
    async_ids = [[] for _ in range(B)]
    for batch_idx in range(B):
        tokens = [tok async for tok in async_generate(target_driver_with_function_scope, input_list[batch_idx], sampling=None, max_new_tokens=MAX_NEW_TOKEN_NUM)]
        async_ids[batch_idx] = tokens
        logger.info(f"async output: |{tokenizer.decode(async_ids[batch_idx])}|, len: {len(async_ids[batch_idx])}")
    
    # sync
    sync_ids = generate(target_driver_with_function_scope.engine.model, input_list, max_new_tokens=MAX_NEW_TOKEN_NUM)
    for batch_idx in range(B):
        logger.info(f"sync output: |{tokenizer.decode(sync_ids[batch_idx])}|, len: {len(sync_ids[batch_idx])}")

    assert async_ids == sync_ids


def _test_benchmark(engine: LLMEngine, input_ids:list[list[int]], tok):
    assert engine.scheduler.is_benchmarking

    # clean
    input_ids = [
        i for i in input_ids if len(i) < engine.model.config.cache_len
    ]

    num_reqs = len(input_ids)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name_flag = f'{engine.model.config.device}.{engine.model.config.max_seqs}.{num_reqs}.{time_str}'
    engine.scheduler.log_name_flag = log_name_flag      # pass to `scheduler` for `stats.xxx` log

    logger.info(f"Starting benchmark..., number of requests: {num_reqs}, max_seqs: {engine.model.config.max_seqs}, cache_len: {engine.model.config.cache_len}")
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

    num_output_ids = sum([sum(o) for o in output_ids])
    logger.info(f"Benchmark results: {elapsed} seconds, rate: {num_output_ids/elapsed/1000} r/s")

def test_benchmark_regularly(target_engine_for_regular_benchmarking, batch_for_regular_benchmarking, tokenizer):
    _test_benchmark(target_engine_for_regular_benchmarking, batch_for_regular_benchmarking, tokenizer)

# pytest -x --log-file-level=DEBUG tests/test_engine.py::test_benchmark_sharegpt --cache_len=256 --req_num=512
# excluded from execution from file, only allowed from specified execution.
@pytest.mark.parametrize("B", [
    1, 
    2, 4, 8, 16, 32, 64,
])
def test_benchmark_sharegpt(target_config, shareGPT_batch_for_sharegpt_benchmarking, tokenizer, req_num:int, B:int, cache_len:int):

    model = QwenForCausalLM(target_config, max_seqs=B, cache_len=cache_len)  # overwrite max_seqs and cache_len for benchmarking
    model.config.stat_interval = 60.

    scheduler = StaticScheduler(model.config, is_benchmarking=True)
    scheduler.max_waiting=req_num
    engine = LLMEngine(model, scheduler)

    _test_benchmark(engine, shareGPT_batch_for_sharegpt_benchmarking, tokenizer)
