import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.models.velocity_mlp import VelocityNetwork


class FM_S3(nn.Module):
    """Rectified Flow Matching model."""

    def __init__(self, hidden_dim: int = 64, sigma_min: float = 1e-4):
        """
        Args:
            input_dim: Dimension of input data
            hidden_dim: Hidden dimension of the velocity network
            sigma_min: Minimum noise level
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sigma_min = sigma_min

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

    def loss(self, x: Tensor) -> Tensor:
        """Compute the Flow Matching loss.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Scalar loss value
        """
        # Sample random time steps and noise
        t = torch.rand(x.shape[0], device=x.device).unsqueeze(-1)
        noise = torch.randn_like(x).to(x.device)

        # Compute noisy sample at time t (previously _compute_noisy_sample)
        x_t = (1 - (1 - self.sigma_min) * t) * noise + t * x

        # Compute optimal flow (previously _compute_optimal_flow)
        optimal_flow = x - (1 - self.sigma_min) * noise

        # Get predicted flow
        predicted_flow = self.forward(x_t, t)

        # Compute MSE loss
        return F.mse_loss(predicted_flow, optimal_flow)

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
