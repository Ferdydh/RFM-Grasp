import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from src.core.config import BaseExperimentConfig

from .velocity_mlp import VelocityNetwork


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
