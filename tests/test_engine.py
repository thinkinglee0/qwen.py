import pytest
import logging

from constants import *
from qwen.engine import ServingDriver, LLMEngine
from qwen.engine import async_generate, benchmark

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
        tokens = [tok async for tok in async_generate(tmp_target_driver, input_list[batch_idx], sampling=None, max_new_tokens=MAX_NEW_TOKENS_FOR_TEST)]
        target_output_token_ids[batch_idx] = tokens

    # reference
    original_repetition_penalty = ref_model.generation_config.repetition_penalty
    try:
        ref_model.generation_config.repetition_penalty = REP_PEN_OFF
        ref_output = ref_model.generate(        # tensor output, shape [B, T]
            **encoding,
            max_new_tokens=MAX_NEW_TOKENS_FOR_TEST,
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
        tokens = [tok async for tok in async_generate(tmp_target_driver, input_ids, sampling=None, max_new_tokens=MAX_NEW_TOKENS_FOR_TEST)]
        ids0.append(tokens)

    logger.info(f"start the second turn")
    ids1 = []
    for input_ids in input_list:
        tokens = [tok async for tok in async_generate(tmp_target_driver, input_ids, sampling=None, max_new_tokens=MAX_NEW_TOKENS_FOR_TEST)]
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
    out_no_pressure, _ = benchmark(engine, batch_for_regular_benchmarking, max_new_tokens=MAX_NEW_TOKENS_FOR_TEST)

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
            out_no_pressure = [tok async for tok in async_generate(driver, input_ids, sampling=None, max_new_tokens=MAX_NEW_TOKENS_FOR_TEST)]
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
