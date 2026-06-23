import json, torch
from dataclasses import dataclass
from pathlib import Path
from safetensors.torch import load_file
import dataclasses


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
    bos_token_id: int
    eos_token_id: int
    model_type: str

    # optional
    rope_scaling: dict | None = None

    # derived
    head_dim: int = 0
    
    def __post_init__(self):
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        if self.rope_scaling is None:
            self.rope_scaling = {"rope_type": "default"}

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "ModelConfig":
        with open(Path(model_dir) / "config.json") as f:
            raw: dict = json.load(f)

        # keep only the fields declared on the dataclass; silently drop any extra keys
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in valid})

def load_qwen_weights(config: ModelConfig, model_dir: str | Path, dtype=torch.float32):
    flat = load_file(Path(model_dir) / "model.safetensors", device="cpu")
    # cast to desired dtype
    flat = {k: v.to(dtype) if v.is_floating_point() else v for k, v in flat.items()}

    if config.tie_word_embeddings and "lm_head.weight" not in flat:
        flat["lm_head.weight"] = flat["model.embed_tokens.weight"]

    return flat

if __name__ == "__main__":
    model_dir = "../qwen2.5-0.5b"
    config = ModelConfig.from_pretrained(model_dir)
    print(config)