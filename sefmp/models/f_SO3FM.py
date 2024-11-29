import torch
import torch.nn as nn
from einops import rearrange
from geomstats._backend import _backend_config as _config
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from scipy.spatial.transform import Rotation
from torch import Tensor

_config.DEFAULT_DTYPE = torch.cuda.FloatTensor

class SO3FM(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.so3_group = SpecialOrthogonal(n=3, point_type="matrix")
        self.hidden_dim = hidden_dim
        
        # Neural network for the velocity field
        self.velocity_net = nn.Sequential(
            nn.Linear(9 + 1, hidden_dim),  # 9 for flattened 3x3 matrix + 1 for time
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 9)  # Output is a flattened 3x3 matrix
        )

    def forward(self, input_data: Tensor) -> Tensor:
        """Forward pass through the velocity network.
        
        Args:
            input_data (Tensor): Concatenated state and time [batch, 10]
            
        Returns:
            Tensor: Velocity prediction in tangent space
        """
        v = self.velocity_net(input_data)
        x = rearrange(input_data[:, :-1], "b (c d) -> b c d", c=3, d=3)
        v = rearrange(v, "b (c d) -> b c d", c=3, d=3)
        Pv = self._tangent_space_proj(x, v)
        return rearrange(Pv, "b c d -> b (c d)", c=3, d=3)

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

    def loss(self, x: Tensor) -> Tensor:
        """Compute training loss.
        
        Args:
            x (Tensor): Target rotation matrices [batch, 3, 3]
            
        Returns:
            Tensor: Scalar loss value
        """
        x1 = x.double()
        x0 = torch.tensor(
            Rotation.random(x1.size(0)).as_matrix(), dtype=torch.float64
        ).to(x1.device)

        # Sample and compute flows
        t, xt, ut = self._sample_location_and_flow(x0, x1)
        
        # Get velocity prediction
        vt = self.forward(
            torch.cat([rearrange(xt, "b c d -> b (c d)", c=3, d=3), t[:, None]], dim=-1)
        )
        vt = rearrange(vt, "b (c d) -> b c d", c=3, d=3)

        return self._loss_so3(vt, ut, xt)

    def _compute_flow(self, t: Tensor, xt: Tensor) -> Tensor:
        """Compute conditional flow."""
        xt = rearrange(xt, "b c d -> b (c d)", c=3, d=3)

        def index_time_der(i):
            return torch.autograd.grad(xt, t, i, create_graph=True, retain_graph=True)[0]

        xt_dot = torch.vmap(index_time_der, in_dims=1)(
            torch.eye(9).to(xt.device).repeat(xt.shape[0], 1, 1)
        )
        return rearrange(xt_dot, "(c d) b -> b c d", c=3, d=3)

    def _sample_location_and_flow(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample location and compute conditional flow."""
        t = torch.rand(x0.shape[0]).type_as(x0).to(x0.device)
        t.requires_grad = True
        
        # Get rotation vector representations
        rot_x0 = self._rotmat_to_rotvec(x0)
        rot_x1 = self._rotmat_to_rotvec(x1)

        # Compute log map
        log_x1 = self.so3_group.log_not_from_identity(rot_x1, rot_x0)
        
        # Sample along geodesic
        xt = self.so3_group.exp_not_from_identity(t.reshape(-1, 1) * log_x1, rot_x0)
        xt = self.so3_group.matrix_from_rotation_vector(xt)
        
        # Compute flow
        ut = self._compute_flow(t, xt)
        
        return t, xt, ut

    def _rotmat_to_rotvec(self, matrix: Tensor) -> Tensor:
        """Convert rotation matrix to rotation vector."""
        # This is a simplified version - you might want to implement a more robust conversion
        rotation = Rotation.from_matrix(matrix.cpu().numpy())
        return torch.tensor(rotation.as_rotvec(), device=matrix.device)

    def inference(self, xt: Tensor, t: Tensor, dt: Tensor) -> Tensor:
        """Single step inference.
        
        Args:
            xt (Tensor): Current state
            t (Tensor): Current time
            dt (Tensor): Time step
            
        Returns:
            Tensor: Next state prediction
        """
        with torch.no_grad():
            vt = self.forward(torch.cat([xt, t[:, None]], dim=-1))
            vt = rearrange(vt, "b (c d) -> b c d", c=3, d=3)
            xt = rearrange(xt, "b (c d) -> b c d", c=3, d=3)
            xt_new = self._expmap(xt, vt * dt)
        return rearrange(xt_new, "b c d -> b (c d)", c=3, d=3)

    def generate(self, x: Tensor, steps: int = 100) -> Tensor:
        """Generate complete trajectory.
        
        Args:
            x (Tensor): Target state tensor
            steps (int, optional): Number of steps. Defaults to 100.
            
        Returns:
            Tensor: Generated final state
        """
        n_test = x.shape[0]
        traj = torch.tensor(Rotation.random(n_test).as_matrix()).reshape(-1, 9)
        
        for t in torch.linspace(0, 1, steps):
            t = torch.tensor([t], dtype=torch.float64).repeat(n_test)
            dt = torch.tensor([1 / steps])
            traj = self.inference(traj, t, dt)
            
        final_traj = rearrange(traj, "b (c d) -> b c d", c=3, d=3)
        return final_traj[-1]