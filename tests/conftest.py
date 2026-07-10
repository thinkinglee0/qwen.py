import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import *
from qwen.config import ModelConfig
from qwen.model import QwenForCausalLM


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)

@pytest.fixture(scope="session")
def inputs(tokenizer):
    return tokenizer(CLASSIC_PROMPT, return_tensors="pt")    # transformers.tokenization_utils_base.BatchEncoding {input_ids, attention_mask}


# my implementation
@pytest.fixture(scope="session")
def target_config():
    return ModelConfig.from_pretrained(MODEL_DIR)       # load weights

@pytest.fixture(scope="session")
def target_model(target_config):
    return QwenForCausalLM(target_config)

@pytest.fixture(scope="function")
def target_model_with_function_scope(target_config):
    return QwenForCausalLM(target_config)

# instance of modeling_qwen2.py from transformers
@pytest.fixture(scope="session")
def ref_model():
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, attn_implementation="eager")
    ref_model.eval()
    return ref_model

