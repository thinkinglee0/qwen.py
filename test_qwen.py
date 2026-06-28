import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from qwen import QwenModel, make_causal_mask
from utils import resolve_device, default_dtype


logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="module")
def inputs(tokenizer):
    input_str = "The capital of France is"
    return tokenizer(input_str, return_tensors="pt")    # transformers.tokenization_utils_base.BatchEncoding {input_ids, attention_mask}

# my implementation
@pytest.fixture(scope="module")
def target_model():
    return QwenModel(MODEL_DIR)

# instance of modeling_qwen2.py from transformers
@pytest.fixture(scope="module")
def ref_model():
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, attn_implementation="eager")
    ref_model.eval()
    return ref_model


def sampling(tok, last_logit):
    new_token_id = last_logit.argmax()
    return tok.decode(new_token_id.item()), new_token_id

# constant variables
MODEL_DIR = "../qwen2.5-0.5b"
MAX_NEW_TOKEN_NUM = 40

# map[platform:xx]
CUR_HOST_PLATFORM = "mac"
EXPECTED_DEVICE_DTYPE_MAP = {
    "mac": {"device": torch.device("cpu"), "dtype": torch.float32},
    "a10": {"device": torch.device("cuda"), "dtype": torch.bfloat16},
}

def get_expected_value(key: str):
    return EXPECTED_DEVICE_DTYPE_MAP[CUR_HOST_PLATFORM][key]


def test_device_dtype():
    device = resolve_device()
    dtype = default_dtype(device)
    assert device == get_expected_value("device")
    assert dtype == get_expected_value("dtype")

def test_causal_mask():
    m = float("-inf")   # masked
    z = 0.0             # zero

    mask = make_causal_mask(2, 2, 
                            device=get_expected_value("device"), 
                            dtype=get_expected_value("dtype"))
    expected = torch.tensor([[[
        [z, m],
        [z, z]]]])
    assert mask.shape == expected.shape
    torch.testing.assert_close(mask, expected)

    mask = make_causal_mask(3, 3, 
                            device=get_expected_value("device"), 
                            dtype=get_expected_value("dtype"))
    expected = torch.tensor([[[
        [z, m, m],
        [z, z, m],
        [z, z, z]]]])
    assert mask.shape == expected.shape
    torch.testing.assert_close(mask, expected)

    mask = make_causal_mask(1, 2, 
                            device=get_expected_value("device"), 
                            dtype=get_expected_value("dtype"))
    # expected = torch.zeros(1, 1, 1, 2)
    expected = torch.tensor([[[
        [z, z]]]])
    assert mask.shape == expected.shape
    torch.testing.assert_close(mask, expected)

    mask = make_causal_mask(2, 3, 
                            device=get_expected_value("device"), 
                            dtype=get_expected_value("dtype"))
    expected = torch.tensor([[[
        [z, z, m],
        [z, z, z]]]])
    assert mask.shape == expected.shape
    torch.testing.assert_close(mask, expected)

# prefill(S+P) == prefill(P) + decode(range(P, S))?
@pytest.mark.parametrize("use_cache", [True, False])
def test_forward_and_kv_cache_correctness(target_model, use_cache: bool):
    torch.manual_seed(0)        # ramdom seed, to fix the executing process.
    S, P = 100, 5
    ids = torch.randint(0, target_model.config.vocab_size, (1, S))

    # sample 1: prefill
    full = target_model.forward(ids, cache=None).logits         # [1, S, V], no cache

    # sample 2: prefill + decode
    output = target_model.forward(ids[:, :P], cache=None)       # prefill P tokens
    cache = output.past_key_values
    incremental = [output.logits[:, -1, :]]     # logit at position P-1
    for t in range(P, S):                       # decode the rest 1-by-1
        if use_cache:
            output = target_model.forward(ids[:, t:t+1], cache=list(cache))
            cache = output.past_key_values
        else:
            output = target_model.forward(ids[:, :t+1], cache=None)
        
        incremental.append(output.logits[:, -1, :])

    # shape [bsz, seq_len, d]
    full_slice = torch.stack([full[:, p, :] for p in range(P-1, S)], 1)[0]
    inc_stack  = torch.cat(incremental, 0)
    diff = (full_slice - inc_stack).abs()
    logger.info(f"max abs logit diff: {diff.max().item():e}")
    assert diff.max().item() < 1e-3          # safe upper bound of FP32 floor
    assert (full_slice.argmax(-1) != inc_stack.argmax(-1)).sum() == 0

# target_model == ref_model?
def test_forward_comparason_with_reference_on_math(target_model, ref_model):
    torch.manual_seed(0)        # ramdom seed, to fix the executing process.
    S, P = 100, 5
    input_ids = torch.randint(0, target_model.config.vocab_size, (1, P))

    for index in range(P, S):
        target_output = target_model.forward(input_ids)
        ref_output = ref_model.forward(input_ids)
        new_token_id = target_output.logits[0, -1].argmax()

        torch.testing.assert_close(target_output.logits, ref_output.logits, rtol=1e-3, atol=1e-3,
                                   msg=lambda s: f"logits mismatch, index={index}\n{s}")

        # shape [bsz, seq_len]
        input_ids = torch.cat([input_ids, torch.full((1,1), new_token_id)], -1)

# fixed inputs, (e.g.: The capital of France is)
def test_forward_comparason_with_reference(target_model, ref_model, tokenizer, inputs):
    input_ids = inputs.input_ids.detach().clone()
    for index in range(MAX_NEW_TOKEN_NUM):
        target_output = target_model.forward(input_ids)
        ref_output = ref_model.forward(input_ids)
        target_new_token, target_new_token_id = sampling(tokenizer, target_output.logits[0, -1])
        ref_new_token, ref_new_token_id = sampling(tokenizer, ref_output.logits[0, -1])
        logger.info(f"index: {index}, target: {target_new_token_id} - |{target_new_token}| <-> ref: {ref_new_token_id} - |{ref_new_token}|")

        torch.testing.assert_close(target_output.logits, ref_output.logits, rtol=1e-3, atol=1e-3,
                                   msg=lambda s: f"logits mismatch, index={index}\n{s}")

        # shape [bsz, seq_len]
        input_ids = torch.cat([input_ids, torch.full((1,1), target_new_token_id)], -1)

REPETITION_PENALTY_SWITCH_OFF = 1.0
REPETITION_PENALTY_DEFAULT = 1.1
@pytest.mark.parametrize("repetition_penalty", [REPETITION_PENALTY_SWITCH_OFF, REPETITION_PENALTY_DEFAULT])
def test_generation_comparason_with_reference(target_model, ref_model, tokenizer, inputs, repetition_penalty: float):
    logger.info(f"input_ids: {inputs.input_ids.shape}")
    target_output_token_ids = target_model.generate(inputs.input_ids, MAX_NEW_TOKEN_NUM)
    target_output_tokens = tokenizer.decode(target_output_token_ids[0])
    logger.info(f"target_output_tokens: |{target_output_tokens}|")

    ref_model.generation_config.repetition_penalty = repetition_penalty
    ref_output = ref_model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKEN_NUM,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        num_beams=1,
    )
    ref_output_tokens = tokenizer.decode(ref_output[0])   # bsz=0
    logger.info(f"   ref_output_tokens: |{ref_output_tokens}|")
    if target_output_tokens != ref_output_tokens:
        logger.info("comparison result: differ")
    else:
        logger.info("comparison result: match")
    
    if repetition_penalty == REPETITION_PENALTY_SWITCH_OFF:
        assert target_output_tokens == ref_output_tokens
    else:
        assert target_output_tokens != ref_output_tokens


