import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import dataclasses

from constants import *
from qwen.config import ModelConfig
from qwen.model import QwenForCausalLM
from qwen.constants import MODEL_DIR
from qwen.engine import ServingDriver, LLMEngine
from qwen.scheduler import Scheduler
from qwen.cache import KVCache
from qwen.utils import sample_sharegpt
from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Enforce custom module execution order, independent of filename sorting.
MODULE_ORDER = ["test_rope", "test_sampling", "test_cache", "test_attention", "test_model", "test_scheduler", "test_engine", "test_api"]

def pytest_collection_modifyitems(session, config, items):
    excluded_fun_names = ["test_benchmark_sharegpt"]
    excluded_module_names = ["test_playground"]

    explicitly_called = any(fun_name in arg for fun_name in excluded_fun_names for arg in config.args)
    explicitly_called |= any(module_name in arg for module_name in excluded_module_names for arg in config.args)

    if not explicitly_called:
        selected = []
        deselected = []
        
        for item in items:
            module_name = item.module.__name__.rsplit(".", 1)[-1]
            fun_name = getattr(item, "originalname", None) or item.name
            if fun_name in excluded_fun_names or module_name in excluded_module_names:
                deselected.append(item)
            else:
                selected.append(item)
                
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = selected
    
    def sort_key(item):
        module_name = item.module.__name__.rsplit(".", 1)[-1]
        try:
            return MODULE_ORDER.index(module_name)
        except ValueError:
            return len(MODULE_ORDER)  # unlisted modules go last

    # list.sort is stable, so unlisted modules keep their original order
    items.sort(key=sort_key)


# options
def pytest_addoption(parser):
    parser.addoption(
        "--req_num",
        action="store",
        default=SHARE_GPT_REQ_NUM,
        type=int,
        help="The number of requests for benchmark (e.g.: 128)"
    )

    parser.addoption(
        "--max_num_seqs",
        action="store",
        default=SHARE_GPT_MAX_SEQS,
        type=int,
        help="The maximum number of sequences in scheduling for benchmark (e.g.: 8)"
    )

    parser.addoption(
        "--max_model_len",
        action="store",
        default=MAX_MODEL_LEN,
        type=int,
        help="The maximum length of kv cache for benchmark (e.g.: 512)"
    )

@pytest.fixture(scope="session")
def req_num(request) -> int:
    return request.config.getoption("--req_num")

@pytest.fixture(scope="session")
def max_num_seqs(request) -> int:
    return request.config.getoption("--max_num_seqs")

@pytest.fixture(scope="session")
def max_model_len(request) -> int:
    return request.config.getoption("--max_model_len")


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
def solo_long_input_ids_tensor(solo_long_encoding) -> torch.Tensor:
    return solo_long_encoding.input_ids     # shape [bsz, seq_len]

@pytest.fixture(scope="function")
def solo_long_input_ids_list(tokenizer) -> list[list[int]]:
    return tokenizer(LONG_BATCH).input_ids

@pytest.fixture(scope="function")
def batch_for_regular_benchmarking(tokenizer) -> list[list[int]]:
    return tokenizer(BATCH_FOR_BENCHMARKING).input_ids


# my implementation
@pytest.fixture(scope="session")
def target_config():
    config = ModelConfig.from_pretrained(MODEL_DIR)       # load weights
    config.num_blocks = 512
    config.cache_verification_interval = 1. 
    return config

@pytest.fixture(scope="function")
def tmp_target_config(target_config):
    tmp = dataclasses.replace(target_config, weights=None)
    tmp.weights = target_config.weights
    return tmp

@pytest.fixture(scope="session")
def target_model(target_config):
    return QwenForCausalLM(target_config)

@pytest.fixture(scope="session")
def target_driver(target_config) -> Iterator[ServingDriver]:
    engine = LLMEngine(target_config)
    with ServingDriver(engine) as d:
        yield d

@pytest.fixture(scope="function")
def tmp_cache(tmp_target_config: ModelConfig):
    return KVCache(tmp_target_config)

@pytest.fixture(scope="function")
def tmp_target_driver(tmp_target_config) -> Iterator[ServingDriver]:
    engine = LLMEngine(tmp_target_config)
    with ServingDriver(engine) as d:
        yield d

# instance of modeling_qwen2.py from transformers
@pytest.fixture(scope="session")
def ref_model():
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, attn_implementation="eager")
    ref_model.eval()
    return ref_model


# benchmark
@pytest.fixture(scope="function")
def shareGPT_batch_for_sharegpt_benchmarking(tokenizer, req_num, max_model_len) -> list[list[int]]:
    return sample_sharegpt(SHARE_GPT_FILE_NAME, tokenizer,  num_requests=req_num, max_p_len=max_model_len//2, max_model_len=max_model_len)


@pytest.fixture(scope="function")
def target_engine_for_regular_benchmarking(tmp_target_config: ModelConfig) -> LLMEngine:
    # overwrite max_num_seqs and max_model_len for benchmarking
    tmp_target_config.max_num_seqs = 4
    tmp_target_config.max_model_len = 128
    tmp_target_config.stat_interval = 10.
    tmp_target_config.is_benchmarking = True
    tmp_target_config.max_waiting=100

    return LLMEngine(tmp_target_config)

@pytest.fixture(scope="function")
def target_engine_for_sharegpt_benchmarking(tmp_target_config, req_num:int, max_num_seqs:int, max_model_len:int) -> LLMEngine:
    # overwrite max_num_seqs and max_model_len for benchmarking
    tmp_target_config.max_num_seqs = max_num_seqs
    tmp_target_config.max_model_len = max_model_len
    tmp_target_config.stat_interval = 60.
    tmp_target_config.is_benchmarking = True
    tmp_target_config.max_waiting=req_num

    return LLMEngine(tmp_target_config)


