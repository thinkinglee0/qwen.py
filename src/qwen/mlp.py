import torch
from torch import nn

from qwen.config import ModelConfig

class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_index: int):
        super().__init__()
        self.weights = cfg.weights
        self.layer_index = layer_index

    def forward(self, hidden_states):
        # mlp
        gate = hidden_states @ self.weights[f'model.layers.{self.layer_index}.mlp.gate_proj.weight'].transpose(-2, -1)
        up = hidden_states @ self.weights[f'model.layers.{self.layer_index}.mlp.up_proj.weight'].transpose(-2, -1)
        mlp_act = torch.nn.functional.silu(gate) * up
        hidden_states = mlp_act @ self.weights[f'model.layers.{self.layer_index}.mlp.down_proj.weight'].transpose(-2, -1)

        return hidden_states