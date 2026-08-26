# src/config.py

import orjson, torch
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path
from safetensors.torch import load_file
import dataclasses
import logging
from collections.abc import Iterable

from qwen.utils import resolve_device, default_dtype
from qwen.constants import LOG_DIR

logger = logging.getLogger(__name__)

def _normalize_eos(value: int | Iterable[int] | None) -> frozenset[int]:
    """HF configs expose eos_token_id as int, list[int], or None."""
    if value is None:
        return frozenset()
    if isinstance(value, int):  # note: bool is a subclass of int, harmless here
        return frozenset((value,))
    return frozenset(value)

@dataclass
class ModelConfig():
    architectures: list[str]
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_act: str
    max_position_embeddings: int
    initializer_range: float
    rms_norm_eps: float
    use_cache: bool
    tie_word_embeddings: bool
    rope_theta: float
    use_sliding_window: bool
    sliding_window: int
    max_window_layers: int
    attention_dropout: float
    torch_dtype: str
    transformers_version: str
    model_type: str

    # from generation_config.json
    do_sample: bool
    bos_token_id: int
    pad_token_id: int
    eos_token_id: list[int] | int | None

    # optional from config.json
    rope_scaling: dict | None = None

    # optional from generation_config.json
    temperature: float = 1.
    top_k: int = 0
    top_p: float = 1.
    do_penalities: bool = True
    repetition_penalty: float = 1.
    frequency_penalty: float = 0.
    presence_penalty: float = 0.

    # derived
    head_dim: int = 0
    eos_token_id_set: frozenset[int] = field(init=False)

    # other
    model_dir: str = ""
    weights: Any | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    # continuous batching
    use_d_first_schedule: bool=True         # D_first_preemptive_schedule if True else preemptive_schedule
    max_model_len: int = 512                # todo: find a suitable value.
    max_num_batched_tokens: int = 1024      # idem
    long_prefill_token_threshold: int = 256 # idem
    max_num_seqs: int = 128                 # batch size; idem
    max_waiting: int = 64                   # idem

    # paged cache
    num_blocks: int = 1024*2   # 2 * 24 * 1024*2 * 256 * 2 * 64 * 2 B = 6442450944 B ≈ 6.4 GB
    block_size: int = 256

    # backoff after preempted
    backoff_base: int = 2
    backoff_cap: int = 64

    # timing tasks
    is_benchmarking: bool=False
    req_metrics_interval: float = 60               # sec
    cache_verification_interval: float = 60 # sec

    # log
    log_dir: str = LOG_DIR

    def __post_init__(self):
        assert self.block_size % 256 == 0, f"flash-attn paged KV requires block_size % 256 == 0, got {self.block_size}"

        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        if self.rope_scaling is None:
            self.rope_scaling = {"rope_type": "default"}

        if self.device is None:
            self.device = resolve_device()
        if self.dtype is None:
            self.dtype = default_dtype(self.device)

        assert self.max_model_len <= self.max_position_embeddings

        self.eos_token_id_set = _normalize_eos(self.eos_token_id)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "ModelConfig":
        with open(Path(model_dir) / "config.json") as f:
            raw = orjson.loads(f.read())

        with open(Path(model_dir) / "generation_config.json") as f:
            raw2 = orjson.loads(f.read())

        raw.update(raw2)    # merge generation config into model config
        raw["model_dir"] = model_dir    # inject

        # keep only the fields declared on the dataclass; silently drop any extra keys
        valid = {f.name for f in dataclasses.fields(cls)}
        config = cls(**{k: v for k, v in raw.items() if k in valid})

        logger.info(f"config: {config}")

        assert config.dtype in [torch.float16, torch.bfloat16, torch.float32], f"unsupported dtype: {config.dtype}"
        config.weights = load_qwen_weights(config, config.dtype)

        return config

def load_qwen_weights(config: ModelConfig, dtype=torch.float32):
    logger.info(f"load_qwen_weights, dir: {config.model_dir}")
    flat = load_file(Path(config.model_dir) / "model.safetensors", device="cpu")
    # cast to desired dtype
    flat = {k: v.to(dtype) if v.is_floating_point() else v for k, v in flat.items()}
    logger.info(f"load_qwen_weights finished")

    return flat

if __name__ == "__main__":
    model_dir = "../qwen2.5-0.5b"
    config = ModelConfig.from_pretrained(model_dir)
    assert config.weights
    print(config.weights.keys())