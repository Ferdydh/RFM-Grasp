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
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for time dimension
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass through the velocity network.

        Args:
            x: Input tensor of shape [batch_size, input_dim]
            t: Time tensor of shape [batch_size, 1]

        Returns:
            Predicted velocity of shape [batch_size, input_dim]
        """
        return self.net(torch.cat([x, t], dim=1))
