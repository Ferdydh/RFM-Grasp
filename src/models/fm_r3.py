import torch
import torch.nn as nn
from torch import Tensor
from src.models.velocity_mlp import VelocityNetwork


class FM_R3(nn.Module):
    """Rectified Flow Matching model."""

    def __init__(self, hidden_dim: int = 64):
        """
        Args:
            input_dim: Dimension of input data
            hidden_dim: Hidden dimension of the velocity network
            sigma_min: Minimum noise level
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # 3 for R3
        self.velocity_net = VelocityNetwork(3, hidden_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute velocity field for given points and times.

        Args:
            x: Input tensor of shape [batch_size, input_dim]
            t: Time tensor of shape [batch_size, 1]

        Returns:
            Predicted velocity field of shape [batch_size, input_dim]
        """
        return self.velocity_net(x, t)

    @torch.no_grad()
    def inference_step(self, x: Tensor, t: Tensor, dt: Tensor) -> Tensor:
        """Perform a single inference step.

        Args:
            x: Current state tensor of shape [batch_size, input_dim]
            t: Current time tensor of shape [batch_size, 1]
            dt: Time step size tensor

        Returns:
            Next state prediction
        """
        dx_dt = self.forward(x, t)
        return x + dt * dx_dt
