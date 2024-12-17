import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from .so3_condflowmatcher import SO3ConditionalFlowMatcher
from scipy.spatial.transform import Rotation
from .velocity_mlp import VelocityNetwork


class SO3FM(nn.Module):
    """Flow Matching model for SO(3) manifold."""

    def __init__(self, hidden_dim: int = 64):
        """
        Args:
            hidden_dim: Hidden dimension of the velocity network
        """
        super().__init__()
        self.so3_group = SpecialOrthogonal(n=3, point_type="matrix")
        self.so3_cfm = SO3ConditionalFlowMatcher(manifold=self.so3_group)
        self.hidden_dim = hidden_dim

        # Use shared VelocityNetwork with SELU activation for SO3
        self.velocity_net = VelocityNetwork(
            input_dim=9,  # Flattened 3x3 matrix
            hidden_dim=hidden_dim,
            activation=nn.SELU,
        )

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
        Pv = self._tangent_space_proj(x, v)
        return rearrange(Pv, "b c d -> b (c d)", c=3, d=3)

    def loss(self, x: Tensor) -> Tensor:
        """Compute training loss.

        Args:
            x: Target rotation matrices [batch, 3, 3]

        Returns:
            Scalar loss value
        """
        x1 = x
        x0 = torch.tensor(Rotation.random(x1.size(0)).as_matrix()).to(x1.device)
        t, xt, ut = self.so3_cfm.sample_location_and_conditional_flow_simple(x0, x1)

        # Get velocity prediction
        xt_flat = rearrange(xt, "b c d -> b (c d)", c=3, d=3)
        vt = self.forward(xt_flat, t[:, None])
        vt = rearrange(vt, "b (c d) -> b c d", c=3, d=3)

        return self._loss_so3(vt, ut, xt)

    def _exp(self, A: Tensor) -> Tensor:
        """Matrix exponential."""
        return torch.linalg.matrix_exp(A)

    def _expmap(self, R0: Tensor, tangent: Tensor) -> Tensor:
        """Exponential map from tangent space to manifold."""
        skew_sym = torch.einsum("...ij,...ik->...jk", R0, tangent)
        return torch.einsum("...ij,...jk->...ik", R0, self._exp(skew_sym))

    def _pt_to_identity(self, R: Tensor, v: Tensor) -> Tensor:
        """Parallel transport to identity."""
        return torch.transpose(R, dim0=-2, dim1=-1) @ v

    def _norm_SO3(self, R: Tensor, T_R: Tensor) -> Tensor:
        """Compute norm in SO(3)."""
        r = self._pt_to_identity(R, T_R)
        return -torch.diagonal(r @ r, dim1=-2, dim2=-1).sum(dim=-1) / 2

    def _tangent_space_proj(self, R: Tensor, M: Tensor) -> Tensor:
        """Project matrix M onto tangent space of SO(3) at point R."""
        skew_symmetric_part = 0.5 * (M - M.permute(0, 2, 1))
        return R @ skew_symmetric_part

    def _loss_so3(self, v: Tensor, u: Tensor, x: Tensor) -> Tensor:
        """Compute SO(3) specific loss."""
        res = v - u
        norm = self._norm_SO3(x, res)
        return torch.mean(norm, dim=-1)

    def _compute_flow(self, t: Tensor, xt: Tensor) -> Tensor:
        """Compute conditional flow.

        Args:
            t: Time tensor
            xt: State tensor [batch, 9]

        Returns:
            Flow field [batch, 3, 3]
        """
        xt = rearrange(xt, "b c d -> b (c d)", c=3, d=3)

        def index_time_der(i: Tensor) -> Tensor:
            return torch.autograd.grad(xt, t, i, create_graph=True, retain_graph=True)[
                0
            ]

        xt_dot = torch.vmap(index_time_der, in_dims=1)(
            torch.eye(9).to(xt.device).repeat(xt.shape[0], 1, 1)
        )
        return rearrange(xt_dot, "(c d) b -> b c d", c=3, d=3)

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
        xt_new = self._expmap(xt, vt * dt)
        return rearrange(xt_new, "b c d -> b (c d)", c=3, d=3)

    @torch.no_grad()
    def generate(self, x: Tensor, steps: int = 100) -> Tensor:
        """Generate complete trajectory.

        Args:
            x: Target state tensor [batch, 3, 3]
            steps: Number of integration steps

        Returns:
            Generated final state [3, 3]
        """
        n_test = x.shape[0]
        traj = torch.tensor(
            Rotation.random(n_test).as_matrix(), dtype=torch.float64
        ).reshape(-1, 9)

        for t in torch.linspace(0, 1, steps):
            t = torch.tensor([t], dtype=torch.float64).repeat(n_test)
            dt = torch.tensor([1 / steps])
            traj = self.inference(traj, t, dt)

        final_traj = rearrange(traj, "b (c d) -> b c d", c=3, d=3)
        return final_traj[-1]
