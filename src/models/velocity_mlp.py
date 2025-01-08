from torch import Tensor
import torch.nn as nn

from src.core.config import BaseExperimentConfig


class VelocityNetwork(nn.Module):
    """Neural network for predicting the velocity field."""

    def __init__(self, input_dim: int, config: BaseExperimentConfig):
        super().__init__()

        self.config = config

        # TODO: use config
        hidden_dim: int = 32
        activation = nn.ReLU

        # Time embedding
        self.time_proj = nn.Sequential(nn.Linear(1, hidden_dim), activation())

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
        )

        # Output projection
        self.final = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # Ensure x is 2D [batch, dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Ensure t is 2D [batch, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        elif t.dim() == 3:
            t = t.squeeze(1)

        # Process time and state
        t_emb = self.time_proj(t)
        h = self.input_proj(x)

        # Combine time embedding and state
        h = h + t_emb

        # Pass through hidden layers
        h = self.hidden_layers(h)

        # Final output
        return self.final(h)
