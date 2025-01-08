from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from scipy.spatial.transform import Rotation
from einops import rearrange

from src.core.config import BaseExperimentConfig

from .velocity_mlp import VelocityNetwork


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
                torch.tensor([t_i], dtype=torch.float64).repeat(num_samples).to(device)
            )
            so3_traj, r3_traj = self.inference_step(so3_traj, r3_traj, t_batch, dt)

        # Reshape SO3 output
        final_so3 = so3_traj.reshape(num_samples, 3, 3)

        return final_so3, r3_traj


class FM_R3(nn.Module):
    """Rectified Flow Matching model."""

    def __init__(self, config: BaseExperimentConfig):
        """
        Args:
            input_dim: Dimension of input data
            hidden_dim: Hidden dimension of the velocity network
            sigma_min: Minimum noise level
        """
        super().__init__()

        self.config = config

        # 3 for R3
        self.velocity_net = VelocityNetwork(3, config)

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


class FM_SO3(nn.Module):
    """Flow Matching model for SO(3) manifold."""

    def __init__(self, config: BaseExperimentConfig):
        """
        Args:
            hidden_dim: Hidden dimension of the velocity network
        """
        super().__init__()

        self.config = config

        # 9 for flattened 3x3 matrix
        self.velocity_net = VelocityNetwork(input_dim=9, config=config)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass to compute velocity field.

        Args:
            x: Input state [batch, 9]
            t: Time [batch, 1]

        Returns:
            Velocity prediction in tangent space [batch, 9]
        """
        v = self.velocity_net(x, t)
        x = rearrange(x, "b (c d) -> b c d", c=3, d=3)
        v = rearrange(v, "b (c d) -> b c d", c=3, d=3)

        # Inline _tangent_space_proj
        skew_symmetric_part = 0.5 * (v - v.permute(0, 2, 1))
        Pv = x @ skew_symmetric_part

        return rearrange(Pv, "b c d -> b (c d)", c=3, d=3)

    @torch.no_grad()
    def inference(self, xt: Tensor, t: Tensor, dt: Tensor) -> Tensor:
        """Single step inference.

        Args:
            xt: Current state [batch, 9]
            t: Current time [batch]
            dt: Time step size [1]

        Returns:
            Next state prediction [batch, 9]
        """
        vt = self.forward(xt, t[:, None])
        vt = rearrange(vt, "b (c d) -> b c d", c=3, d=3)
        xt = rearrange(xt, "b (c d) -> b c d", c=3, d=3)

        # Inline _expmap and _exp
        skew_sym = torch.einsum("...ij,...ik->...jk", xt, vt * dt)
        xt_new = torch.einsum(
            "...ij,...jk->...ik", xt, torch.linalg.matrix_exp(skew_sym)
        )

        return rearrange(xt_new, "b c d -> b (c d)", c=3, d=3)
