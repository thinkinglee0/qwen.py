import pytest
import torch
import logging

from qwen.engine import async_generate, generate
from constants import MAX_NEW_TOKEN_NUM


logger = logging.getLogger(__name__)


def test_generation_compared_with_reference(target_model_with_function_scope, ref_model, tokenizer, inputs):
    rep_pen_off = 1.
    temp_greedy = 0.

    target_model_with_function_scope.config.repetition_penalty = rep_pen_off
    target_model_with_function_scope.config.temperature = temp_greedy
    target_output_token_ids = generate(target_model_with_function_scope, inputs.input_ids, max_new_tokens=MAX_NEW_TOKEN_NUM)
    target_output_tokens = tokenizer.decode(target_output_token_ids[0])
    logger.info(f"target_output_tokens: |{target_output_tokens}|")

    try:
        original_repetition_penalty = ref_model.generation_config.repetition_penalty
        ref_model.generation_config.repetition_penalty = rep_pen_off
        ref_output = ref_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKEN_NUM,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            num_beams=1,
        )
    finally:
        ref_model.generation_config.repetition_penalty = original_repetition_penalty

    ref_output_tokens = tokenizer.decode(ref_output[0])   # bsz=0
    logger.info(f"   ref_output_tokens: |{ref_output_tokens}|")
    if target_output_tokens != ref_output_tokens:
        logger.info("comparison result: differ")
    else:
        logger.info("comparison result: match")
    
    assert target_output_tokens == ref_output_tokens

@pytest.mark.asyncio
async def test_streaming_generation(target_model, tokenizer, inputs):
    # initialize temperature with greedy
    bsz, _ = inputs.input_ids.size()
    temperature = torch.full(
        size=(bsz,), 
        fill_value=0.,
        device=target_model.device,
        dtype=target_model.dtype
    )

    # async
    stream_chunks = [tok async for tok in async_generate(target_model, inputs.input_ids, temperature=temperature, max_new_tokens=MAX_NEW_TOKEN_NUM)]
    async_ids = torch.cat([t.reshape(t.shape[0], -1) for t in stream_chunks], dim=1)
    logger.info(f"async output: |{tokenizer.decode(async_ids[0])}|")
    
    # sync
    sync_ids = generate(target_model, inputs.input_ids, temperature=temperature, max_new_tokens=MAX_NEW_TOKEN_NUM)
    logger.info(f" sync output: |{tokenizer.decode(sync_ids[0])}|")

    assert torch.equal(async_ids, sync_ids)

def test_generations_differentiation(target_model, tokenizer, inputs):
    sync_ids0 = generate(target_model, inputs.input_ids, max_new_tokens=MAX_NEW_TOKEN_NUM)
    logger.info(f"output-0: |{tokenizer.decode(sync_ids0[0])}|")


    sync_ids1 = generate(target_model, inputs.input_ids, max_new_tokens=MAX_NEW_TOKEN_NUM)
    logger.info(f"output-1: |{tokenizer.decode(sync_ids1[0])}|")

    assert not torch.equal(sync_ids0, sync_ids1)
