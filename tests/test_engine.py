import pytest
import torch
import logging

from qwen.engine import async_generate, generate
from constants import MAX_NEW_TOKEN_NUM


logger = logging.getLogger(__name__)

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

    rep_pen_off = 1.
    temp_greedy = 0.

    target_model_with_function_scope.config.repetition_penalty = rep_pen_off
    target_model_with_function_scope.config.temperature = temp_greedy
    target_output_token_ids = generate(target_model_with_function_scope, input_list, max_new_tokens=MAX_NEW_TOKEN_NUM)

    try:
        original_repetition_penalty = ref_model.generation_config.repetition_penalty
        ref_model.generation_config.repetition_penalty = rep_pen_off
        ref_output = ref_model.generate(        # tensor output, shape [B, T]
            **encoding,
            max_new_tokens=MAX_NEW_TOKEN_NUM,
            do_sample=False, temperature=None,
            top_p=None, top_k=None, num_beams=1,
        )
    finally:
        ref_model.generation_config.repetition_penalty = original_repetition_penalty

    # compare with reference model
    logger.info(f"attention_mask: {encoding.attention_mask}")
    padding_count = (encoding.attention_mask == 0).sum(dim=1).tolist()      # [B]
    for batch_idx in range(B):
        ref_output_text = tokenizer.decode(ref_output[batch_idx, padding_count[batch_idx]:])
        target_output_text = tokenizer.decode(target_output_token_ids[batch_idx])
        logger.info(f"target_output_text[{batch_idx}]: |{target_output_text}|")
        logger.info(f"   ref_output_text[{batch_idx}]: |{ref_output_text}|")
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
async def test_streaming_generation(target_model_with_function_scope, tokenizer, request, list_fixture):
    input_ids_list = request.getfixturevalue(list_fixture)
    B = len(input_ids_list)

    temp_greedy = 0.
    target_model_with_function_scope.config.temperature = temp_greedy

    # async
    stream_chunks = [tok async for tok in async_generate(target_model_with_function_scope, input_ids_list, max_new_tokens=MAX_NEW_TOKEN_NUM)]
    async_ids = [[] for _ in range(B)]
    for chunk in stream_chunks:
        async_ids = [old_id + new_id for old_id, new_id in zip(async_ids, chunk)]
    # async_ids = [old_id + new_id for chunk in stream_chunks for old_id, new_id in zip(async_ids, chunk)]  # error
    for batch_idx in range(B):
        logger.info(f"async output: |{tokenizer.decode(async_ids[batch_idx])}|, len: {len(async_ids[batch_idx])}")
    
    # sync
    sync_ids = generate(target_model_with_function_scope, input_ids_list, max_new_tokens=MAX_NEW_TOKEN_NUM)
    for batch_idx in range(B):
        logger.info(f"sync output: |{tokenizer.decode(sync_ids[batch_idx])}|, len: {len(sync_ids[batch_idx])}")

    assert async_ids == sync_ids

@pytest.mark.parametrize(
    ("list_fixture"),
    [
        pytest.param("solo_input_ids_list", id="single"),
        pytest.param("batch_input_ids_list", id="batch"),
    ],
)
def test_generations_differentiation(target_model, tokenizer, list_fixture, request):
    input_ids_list = request.getfixturevalue(list_fixture)

    sync_ids0 = generate(target_model, input_ids_list, max_new_tokens=MAX_NEW_TOKEN_NUM)
    sync_ids1 = generate(target_model, input_ids_list, max_new_tokens=MAX_NEW_TOKEN_NUM)

    for batch_idx in range(len(input_ids_list)):
        logger.info(f"batch_idx: {batch_idx}, output-0: |{tokenizer.decode(sync_ids0[batch_idx])}|")
        logger.info(f"batch_idx: {batch_idx}, output-1: |{tokenizer.decode(sync_ids1[batch_idx])}|")

    assert sync_ids0 != sync_ids1
