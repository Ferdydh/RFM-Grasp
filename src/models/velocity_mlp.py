from torch import Tensor
import torch
import torch.nn as nn


class VelocityNetwork(nn.Module):
    """Neural network for predicting the velocity field."""

    def __init__(self, input_dim: int, hidden_dim: int, activation=nn.SiLU):
        """
        Args:
            input_dim: Dimension of input state
            hidden_dim: Hidden dimension of the network
            activation: Activation function to use (default: SiLU/Swish)
        """
        super().__init__()
        # Time embedding
        self.time_proj = nn.Sequential(
            nn.Linear(1, hidden_dim), activation(), nn.LayerNorm(hidden_dim)
        )

        # Main network with residual blocks
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Output projection with small initialization
        self.final = nn.Linear(hidden_dim, input_dim)
        # Initialize final layer with small weights for stability
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass through the velocity network.

        Args:
            x: Input tensor of shape [batch_size, input_dim]
            t: Time tensor of shape [batch_size, 1]

        Returns:
            Predicted velocity of shape [batch_size, input_dim]
        """
        # Ensure x is 2D [batch, dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Ensure t is 2D [batch, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        elif t.dim() == 3:
            t = t.squeeze(1)  # Remove middle dimension if [batch, 1, 1]

        t_emb = self.time_proj(t)

        # Project input and add time embedding
        h = self.input_proj(x)
        h = h + t_emb

        # Residual blocks
        h = h + self.block1(h)
        h = h + self.block2(h)

        # Scaled output
        return self.final(h) * self.scale
