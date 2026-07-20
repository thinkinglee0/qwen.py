import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import *
from qwen.config import ModelConfig
from qwen.model import QwenForCausalLM
from qwen.constants import MODEL_DIR


# Enforce custom module execution order, independent of filename sorting.
MODULE_ORDER = ["test_rope", "test_sampling", "test_attention", "test_model", "test_engine", "test_api"]

def pytest_collection_modifyitems(session, config, items):
    def sort_key(item):
        module_name = item.module.__name__.rsplit(".", 1)[-1]
        try:
            return MODULE_ORDER.index(module_name)
        except ValueError:
            return len(MODULE_ORDER)  # unlisted modules go last

    # list.sort is stable, so unlisted modules keep their original order
    items.sort(key=sort_key)


# instances for testing
@pytest.fixture(scope="session")
def tokenizer():    # transformers.tokenization_utils_base.BatchEncoding {input_ids, attention_mask}
    return AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="left")

@pytest.fixture(scope="function")
def solo_encoding(tokenizer):
    return tokenizer(PROMPT_CLASSICAL, padding=True, return_tensors="pt")     # BatchEncoding

@pytest.fixture(scope="function")
def input_ids_tensor(solo_encoding):
    return solo_encoding.input_ids     # shape [bsz, seq_len]

@pytest.fixture(scope="function")
def solo_input_ids_list(tokenizer):
    return tokenizer(PROMPT_BATCH_1).input_ids      # type list[list[int]]

@pytest.fixture(scope="function")
def batch_input_ids_list(tokenizer):
    return tokenizer(PROMPT_BATCH_2).input_ids      # ragged batch, ragged nested lists, list[list[int]]

@pytest.fixture(scope="function")
def batch_encoding(tokenizer):
    return tokenizer(PROMPT_BATCH_2, padding=True, return_tensors="pt")      # padded batch, padded rectangular tensor, BatchEncoding

@pytest.fixture(scope="function")
def batch_input_ids_tensor(batch_encoding):
    return batch_encoding.input_ids      # padded batch, padded rectangular tensor

@pytest.fixture(scope="function")
def long_batch_input_ids_list(tokenizer):
    return tokenizer(LONG_BATCH_2).input_ids      # ragged batch, ragged nested lists, list[list[int]]

@pytest.fixture(scope="function")
def long_batch_encoding(tokenizer):
    return tokenizer(LONG_BATCH_2, padding=True, return_tensors="pt")      # padded batch, padded rectangular tensor, BatchEncoding

@pytest.fixture(scope="function")
def long_batch_input_ids_tensor(long_batch_encoding):
    return long_batch_encoding.input_ids      # padded batch, padded rectangular tensor

@pytest.fixture(scope="function")
def solo_long_encoding(tokenizer):
    return tokenizer(LONG_BATCH, padding=True, return_tensors="pt")     # BatchEncoding

@pytest.fixture(scope="function")
def solo_long_input_ids_tensor(solo_long_encoding):
    return solo_long_encoding.input_ids     # shape [bsz, seq_len]

@pytest.fixture(scope="function")
def solo_long_input_ids_list(tokenizer):
    return tokenizer(LONG_BATCH).input_ids      # type list[list[int]]




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

