import pytest
import torch
import logging

from qwen.engine import async_generate, generate
from constants import MAX_NEW_TOKEN_NUM


logger = logging.getLogger(__name__)


def test_generation_compared_with_reference(target_model_with_function_scope, ref_model, tokenizer, solo_input_ids_list, solo_encoding):
    rep_pen_off = 1.
    temp_greedy = 0.

    target_model_with_function_scope.config.repetition_penalty = rep_pen_off
    target_model_with_function_scope.config.temperature = temp_greedy
    target_output_token_ids = generate(target_model_with_function_scope, solo_input_ids_list, max_new_tokens=MAX_NEW_TOKEN_NUM)
    target_output_text = tokenizer.decode(target_output_token_ids[0])
    logger.info(f"target_output_text: |{target_output_text}|")

    try:
        original_repetition_penalty = ref_model.generation_config.repetition_penalty
        ref_model.generation_config.repetition_penalty = rep_pen_off
        ref_output = ref_model.generate(
            **solo_encoding,
            max_new_tokens=MAX_NEW_TOKEN_NUM,
            do_sample=False, temperature=None,
            top_p=None, top_k=None, num_beams=1,
        )
    finally:
        ref_model.generation_config.repetition_penalty = original_repetition_penalty

    ref_output_text = tokenizer.decode(ref_output[0])   # batch_idx=0
    logger.info(f"   ref_output_text: |{ref_output_text}|")
    if target_output_text != ref_output_text:
        logger.info("comparison result: differ")
    else:
        logger.info("comparison result: match")
    
    assert target_output_text == ref_output_text

@pytest.mark.asyncio
async def test_streaming_generation(target_model_with_function_scope, tokenizer, solo_input_ids_list):
    rep_pen_off = 1.
    temp_greedy = 0.

    target_model_with_function_scope.config.repetition_penalty = rep_pen_off
    target_model_with_function_scope.config.temperature = temp_greedy

    # async
    B = len(solo_input_ids_list)
    logger.info(f"solo_input_ids_list len: {len(solo_input_ids_list[0])}")
    stream_chunks = [tok async for tok in async_generate(target_model_with_function_scope, solo_input_ids_list, max_new_tokens=MAX_NEW_TOKEN_NUM)]
    async_ids = [[] for _ in range(B)]
    for chunk in stream_chunks:
        async_ids = [old_id + new_id for old_id, new_id in zip(async_ids, chunk)]
    # async_ids = [old_id + new_id for chunk in stream_chunks for old_id, new_id in zip(async_ids, chunk)]  # error
    logger.info(f"async output: |{tokenizer.decode(async_ids[0])}|, len: {len(async_ids[0])}")
    
    # sync
    logger.info(f"solo_input_ids_list len: {len(solo_input_ids_list[0])}")
    sync_ids = generate(target_model_with_function_scope, solo_input_ids_list, max_new_tokens=MAX_NEW_TOKEN_NUM)
    logger.info(f" sync output: |{tokenizer.decode(sync_ids[0])}|, len: {len(sync_ids[0])}")

    assert async_ids == sync_ids

def test_generations_differentiation(target_model, tokenizer, solo_input_ids_list):
    sync_ids0 = generate(target_model, solo_input_ids_list, max_new_tokens=MAX_NEW_TOKEN_NUM)
    logger.info(f"output-0: |{tokenizer.decode(sync_ids0[0])}|")

    sync_ids1 = generate(target_model, solo_input_ids_list, max_new_tokens=MAX_NEW_TOKEN_NUM)
    logger.info(f"output-1: |{tokenizer.decode(sync_ids1[0])}|")

    assert sync_ids0 != sync_ids1
