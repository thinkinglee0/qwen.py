import torch
from torch import nn

from qwen.config import ModelConfig

class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, hidden_states):
        # mlp
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        mlp_act = torch.nn.functional.silu(gate) * up
        hidden_states = self.down_proj(mlp_act)

        return hidden_states