from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.models.velocity_mlp import VelocityNetwork


class R3FM(nn.Module):
    """Rectified Flow Matching model."""

    def __init__(
        self, input_dim: int = 3, hidden_dim: int = 64, sigma_min: float = 1e-4
    ):
        """
        Args:
            input_dim: Dimension of input data
            hidden_dim: Hidden dimension of the velocity network
            sigma_min: Minimum noise level
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sigma_min = sigma_min

        self.velocity_net = VelocityNetwork(input_dim, hidden_dim)

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

        # Compute noisy sample at time t
        x_t = self._compute_noisy_sample(x, t, noise)

        # Compute optimal flow
        optimal_flow = self._compute_optimal_flow(x, noise)

        # Get predicted flow
        predicted_flow = self.forward(x_t, t)

        # Compute MSE loss
        return F.mse_loss(predicted_flow, optimal_flow)

    def _compute_noisy_sample(self, x: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Compute noisy sample at time t."""
        return (1 - (1 - self.sigma_min) * t) * noise + t * x

    def _compute_optimal_flow(self, x: Tensor, noise: Tensor) -> Tensor:
        """Compute the optimal flow."""
        return x - (1 - self.sigma_min) * noise

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

    @torch.no_grad()
    def sample(self, x_1: Tensor, steps: int = 200) -> Tensor:
        """Generate samples using the learned flow.

        Args:
            x_1: Target state tensor of shape [batch_size, input_dim]
            steps: Number of integration steps

        Returns:
            Generated samples of shape [batch_size, input_dim]
        """
        traj = torch.randn_like(x_1).to(x_1.device)
        t = torch.linspace(0, 1, steps).to(x_1.device)
        dt = torch.tensor([1 / steps]).to(x_1.device)

        for t_i in t:
            t_i = (
                torch.tensor([t_i])
                .to(x_1.device)
                .repeat(traj.size(0), 1)
                .requires_grad_(True)
            )
            traj = self.inference_step(traj, t_i, dt)

        return traj
