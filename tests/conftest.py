import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from constants import *
from qwen.config import ModelConfig
from qwen.model import QwenForCausalLM
from qwen.constants import MODEL_DIR
from qwen.engine import ServingDriver, LLMEngine
from qwen.scheduler import StaticScheduler
from qwen.utils import sample_sharegpt
from qwen.constants import DEFAULT_CACHE_LEN

logger = logging.getLogger(__name__)

# Enforce custom module execution order, independent of filename sorting.
MODULE_ORDER = ["test_rope", "test_sampling", "test_attention", "test_model", "test_engine", "test_api"]

def pytest_collection_modifyitems(session, config, items):
    excluded_names_from_file = {"test_benchmark_sharegpt"}

    explicitly_called = any(fun_name in arg for fun_name in excluded_names_from_file for arg in config.args)
    if not explicitly_called:
        selected = []
        deselected = []
        
        for item in items:
            if item.name in excluded_names_from_file:
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
        "--max_seqs",
        action="store",
        default=SHARE_GPT_MAX_SEQS,
        type=int,
        help="The maximum of sequences for benchmark (e.g.: 8)"
    )

    parser.addoption(
        "--cache_len",
        action="store",
        default=DEFAULT_CACHE_LEN,
        type=int,
        help="The maximum length of kv cache for benchmark (e.g.: 512)"
    )

@pytest.fixture(scope="session")
def req_num(request) -> int:
    return request.config.getoption("--req_num")

@pytest.fixture(scope="session")
def max_seqs(request) -> int:
    return request.config.getoption("--max_seqs")

@pytest.fixture(scope="session")
def cache_len(request) -> int:
    return request.config.getoption("--cache_len")


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

@pytest.fixture(scope="function")
def batch_for_regular_benchmarking(tokenizer):
    return tokenizer(BATCH_FOR_BENCHMARKING).input_ids      # type list[list[int]]


# my implementation
@pytest.fixture(scope="session")
def target_config():
    return ModelConfig.from_pretrained(MODEL_DIR)       # load weights

@pytest.fixture(scope="session")
def target_model(target_config):
    return QwenForCausalLM(target_config)

@pytest.fixture(scope="session")
def target_scheduler(target_config):
    return StaticScheduler(target_config)

@pytest.fixture(scope="session")
def target_driver(target_model, target_scheduler):
    engine = LLMEngine(target_model, target_scheduler)
    driver = ServingDriver(engine)
    driver.start()
    return driver

@pytest.fixture(scope="function")
def target_model_with_function_scope(target_config):
    return QwenForCausalLM(target_config)

@pytest.fixture(scope="function")
def target_driver_with_function_scope(target_model_with_function_scope):
    scheduler = StaticScheduler(target_model_with_function_scope.config)
    engine = LLMEngine(target_model_with_function_scope, scheduler)
    driver = ServingDriver(engine)
    driver.start()
    return driver


# instance of modeling_qwen2.py from transformers
@pytest.fixture(scope="session")
def ref_model():
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, attn_implementation="eager")
    ref_model.eval()
    return ref_model


# benchmark
@pytest.fixture(scope="function")
def shareGPT_batch_for_sharegpt_benchmarking(tokenizer, req_num, cache_len) -> list[list[int]]:
    return sample_sharegpt(SHARE_GPT_FILE_NAME, tokenizer,  num_requests=req_num, max_p_len=cache_len//2, cache_len=cache_len)


@pytest.fixture(scope="function")
def target_engine_for_regular_benchmarking(target_config) -> LLMEngine:
    model = QwenForCausalLM(target_config, max_seqs=4, cache_len=128)  # overwrite max_seqs and cache_len for benchmarking
    model.config.stat_interval = 10.

    scheduler = StaticScheduler(model.config, is_benchmarking=True)
    scheduler.max_waiting=100
    return LLMEngine(model, scheduler)

@pytest.fixture(scope="function")
def target_engine_for_sharegpt_benchmarking(target_config, req_num:int, max_seqs:int, cache_len:int) -> LLMEngine:
    model = QwenForCausalLM(target_config, max_seqs=max_seqs, cache_len=cache_len)  # overwrite max_seqs and cache_len for benchmarking
    model.config.stat_interval = 60.

    scheduler = StaticScheduler(model.config, is_benchmarking=True)
    scheduler.max_waiting=req_num
    return LLMEngine(model, scheduler)


