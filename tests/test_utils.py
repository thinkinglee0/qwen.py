import pytest
import logging
import dataclasses

from qwen.config import ModelConfig
from utils import parse_env_list_value

logger = logging.getLogger(__name__)

@pytest.mark.parametrize(
    ("env_value", "default_value", "expected"),
    [
        ("1,2,3", [1, 2, 3], [1, 2, 3]),
        ("true,false,true", [True, False, True], [True, False, True]),
        ("alpha,beta,gamma", ["alpha", "beta", "gamma"], ["alpha", "beta", "gamma"]),
    ],
)
def test_parse_env_list_value_parses_supported_scalar_types(monkeypatch, env_value, default_value, expected):
    monkeypatch.setenv("TEST_PARSE_ENV_LIST_VALUE", env_value)
    assert parse_env_list_value("TEST_PARSE_ENV_LIST_VALUE", default_value) == expected


def test_parse_env_list_value_uses_default_when_var_missing(monkeypatch):
    monkeypatch.delenv("TEST_PARSE_ENV_LIST_VALUE", raising=False)
    assert parse_env_list_value("TEST_PARSE_ENV_LIST_VALUE", ["left", "right"]) == ["left", "right"]


def test_config(tmp_target_config_for_sharegpt_benchmarking: ModelConfig, compile_rope:bool):
    logger.info(f"compile_rope: {compile_rope}")
    tmp_target_config_for_sharegpt_benchmarking.compile_rope = compile_rope
    cfg = dataclasses.replace(tmp_target_config_for_sharegpt_benchmarking, weights=None)    # deep copy without weights
    logger.info(f"config to json: {cfg.as_json()}")
