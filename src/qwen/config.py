# src/config.py

import json, torch
from typing import Any
from dataclasses import dataclass
from pathlib import Path
from safetensors.torch import load_file
import dataclasses
import logging


logger = logging.getLogger(__name__)

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
    eos_token_id: list[int]
    top_p: float
    top_k: float

    # optional from config.json
    rope_scaling: dict | None = None

    # optional from generation_config.json
    temperature: float = 1.
    top_k: float = 0.
    top_p: float = 1.
    repetition_penalty: float = 1.
    frequency_penalty: float = 0.
    presence_penalty: float = 0.

    # derived
    head_dim: int = 0

    # other
    model_dir: str = ""
    weights: Any | None = None
    
    def __post_init__(self):
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        if self.rope_scaling is None:
            self.rope_scaling = {"rope_type": "default"}

        logger.info(f"config: {self}")

        self.weights = load_qwen_weights(self)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "ModelConfig":
        with open(Path(model_dir) / "config.json") as f:
            raw: dict = json.load(f)

        with open(Path(model_dir) / "generation_config.json") as f:
            raw2: dict = json.load(f)

        raw.update(raw2)    # merge generation config into model config
        raw["model_dir"] = model_dir    # inject

        # keep only the fields declared on the dataclass; silently drop any extra keys
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in valid})

def load_qwen_weights(config: ModelConfig, dtype=torch.float32):
    logger.info(f"load_qwen_weights, dir: {config.model_dir}")
    flat = load_file(Path(config.model_dir) / "model.safetensors", device="cpu")
    # cast to desired dtype
    flat = {k: v.to(dtype) if v.is_floating_point() else v for k, v in flat.items()}

    if config.tie_word_embeddings and "lm_head.weight" not in flat:
        flat["lm_head.weight"] = flat["model.embed_tokens.weight"]

    return flat

if __name__ == "__main__":
    model_dir = "../qwen2.5-0.5b"
    config = ModelConfig.from_pretrained(model_dir)
    print(config)