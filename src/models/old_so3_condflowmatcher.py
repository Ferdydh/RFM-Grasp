import torch
from einops import rearrange
from functorch import vmap
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


def _normalize_quaternion(quat):
    return quat / torch.norm(quat, dim=-1, keepdim=True)


def rotmat_to_quat(matrix):
    # Ensure matrix is float64 to match quat dtype
    matrix = matrix.to(torch.float64)

    num_rots = matrix.shape[0]
    matrix_diag = torch.diagonal(matrix, dim1=-2, dim2=-1)
    matrix_trace = torch.sum(matrix_diag, dim=-1, keepdim=True)
    decision = torch.cat((matrix_diag, matrix_trace), dim=-1)
    choice = torch.argmax(decision, dim=-1)
    quat = torch.zeros((num_rots, 4), dtype=matrix.dtype, device=matrix.device)

    # Indices where choice is not 3
    not_three_mask = choice != 3
    i = choice[not_three_mask]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[not_three_mask, i] = (
        1 - decision[not_three_mask, 3] + 2 * matrix[not_three_mask, i, i]
    ).to(torch.float64)  # Ensure float64
    quat[not_three_mask, j] = (
        matrix[not_three_mask, j, i] + matrix[not_three_mask, i, j]
    ).to(torch.float64)  # Ensure float64
    quat[not_three_mask, k] = (
        matrix[not_three_mask, k, i] + matrix[not_three_mask, i, k]
    ).to(torch.float64)  # Ensure float64
    quat[not_three_mask, 3] = (
        matrix[not_three_mask, k, j] - matrix[not_three_mask, j, k]
    ).to(torch.float64)  # Ensure float64

    # Indices where choice is 3
    three_mask = ~not_three_mask
    quat[three_mask, 0] = (matrix[three_mask, 2, 1] - matrix[three_mask, 1, 2]).to(
        torch.float64
    )
    quat[three_mask, 1] = (matrix[three_mask, 0, 2] - matrix[three_mask, 2, 0]).to(
        torch.float64
    )
    quat[three_mask, 2] = (matrix[three_mask, 1, 0] - matrix[three_mask, 0, 1]).to(
        torch.float64
    )
    quat[three_mask, 3] = (1 + decision[three_mask, 3]).to(torch.float64)

    return _normalize_quaternion(quat)


def quat_to_rotvec(quat, degrees=False):
    quat = torch.where(quat[..., 3:4] < 0, -quat, quat)
    angle = 2.0 * torch.atan2(torch.norm(quat[..., :3], dim=-1), quat[..., 3])
    angle2 = angle * angle
    small_scale = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
    large_scale = angle / torch.sin(angle / 2 + 1e-6)
    scale = torch.where(angle <= 1e-3, small_scale, large_scale)

    if degrees:
        scale = torch.rad2deg(scale)

    return scale[..., None] * quat[..., :3]


def rotmat_to_rotvec(matrix):
    # Check if matrix has 3 dimensions and last two dimensions have shape 3
    if len(matrix.shape) != 3 or matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")
    return quat_to_rotvec(rotmat_to_quat(matrix))


class SO3ConditionalFlowMatcher:
    def __init__(self, manifold):
        self.sigma = None
        self.manifold = manifold
        self.vec_manifold = SpecialOrthogonal(n=3, point_type="vector")

    def vec_log_map(self, x0, x1):
        # get logmap of x_1 from x_0
        # convert to axis angle to compute logmap efficiently
        rot_x0 = rotmat_to_rotvec(x0)
        rot_x1 = rotmat_to_rotvec(x1)

        log_x1 = self.vec_manifold.log_not_from_identity(rot_x1, rot_x0)
        return log_x1, rot_x0

    def sample_xt(self, x0, x1, t):
        # sample along the geodesic from x0 to x1
        log_x1, rot_x0 = self.vec_log_map(x0, x1)
        # group exponential at x0
        xt = self.vec_manifold.exp_not_from_identity(t.reshape(-1, 1) * log_x1, rot_x0)
        xt = self.vec_manifold.matrix_from_rotation_vector(xt)
        return xt

    def compute_conditional_flow_simple(self, t, xt):
        xt = rearrange(xt, "b c d -> b (c d)", c=3, d=3)

        def index_time_der(i):
            return torch.autograd.grad(xt, t, i, create_graph=True, retain_graph=True)[
                0
            ]

        xt_dot = vmap(index_time_der, in_dims=1)(
            torch.eye(9).to(xt.device).repeat(xt.shape[0], 1, 1)
        )
        return rearrange(xt_dot, "(c d) b -> b c d", c=3, d=3)

    def sample_location_and_conditional_flow_simple(self, x0, x1):
        t = torch.rand(x0.shape[0]).type_as(x0).to(x0.device)
        t.requires_grad = True
        xt = self.sample_xt(x0, x1, t)
        ut = self.compute_conditional_flow_simple(t, xt)

        return t, xt, ut
