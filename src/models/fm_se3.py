from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from scipy.spatial.transform import Rotation

from src.core.config import BaseExperimentConfig
from .fm_r3 import FM_R3
from .fm_so3 import FM_SO3


class FM_SE3(nn.Module):
    """Combined Flow Matching model for SE(3) = SO(3) x  R³."""

    def __init__(self, config: BaseExperimentConfig):
        """
        Args:
            r3_dim: Dimension of R3 component (default: 3)
            hidden_dim: Hidden dimension for both networks (default: 64)
        """
        super().__init__()
        self.r3fm = FM_R3(config)
        self.so3fm = FM_SO3(config)

    def forward(
        self, so3_input: Tensor, r3_input: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass for both SO3 and R3 components.

        Args:
            so3_input: SO3 input tensor [batch, 9] (flattened 3x3 matrices)
            r3_input: R3 input tensor [batch, 3]
            t: Time tensor [batch, 1]

        Returns:
            Tuple of (so3_velocity, r3_velocity)
        """
        so3_velocity = self.so3fm(so3_input, t)
        r3_velocity = self.r3fm(r3_input, t)
        return so3_velocity, r3_velocity

    @torch.no_grad()
    def inference_step(
        self, so3_state: Tensor, r3_state: Tensor, t: Tensor, dt: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Single step inference for both manifolds.

        Args:
            so3_state: Current SO3 state [batch, 9]
            r3_state: Current R3 state [batch, 3]
            t: Current time [batch, 1]
            dt: Time step size [1]

        Returns:
            Tuple of (next_so3_state, next_r3_state)
        """
        so3_next = self.so3fm.inference(so3_state, t, dt)
        r3_next = self.r3fm.inference_step(r3_state, t, dt)
        return so3_next, r3_next

    @torch.no_grad()
    def sample(
        self, device: torch.device, num_samples: int = 1, steps: int = 200
    ) -> Tuple[Tensor, Tensor]:
        """Generate samples for both manifolds.

        Args:
            device: Device to generate samples on
            num_samples: Number of samples to generate
            steps: Number of integration steps

        Returns:
            Tuple of (so3_samples, r3_samples) where:
                so3_samples: [num_samples, 3, 3]
                r3_samples: [num_samples, 3]
        """
        # Initialize random starting points
        so3_traj = (
            torch.tensor(Rotation.random(num_samples).as_matrix(), dtype=torch.float64)
            .reshape(-1, 9)
            .to(device)
        )
        r3_traj = torch.randn(num_samples, 3, dtype=torch.float64).to(device)

        # Setup time steps
        t = torch.linspace(0, 1, steps).to(device)
        dt = torch.tensor([1 / steps]).to(device)

        # Generate trajectories
        for t_i in t:
            t_batch = (
                torch.tensor([t_i], dtype=torch.float64)
                .repeat(num_samples)
                .to(device)
            )
            so3_traj, r3_traj = self.inference_step(so3_traj, r3_traj, t_batch, dt)

        # Reshape SO3 output
        final_so3 = so3_traj.reshape(num_samples, 3, 3)

        return final_so3, r3_traj
