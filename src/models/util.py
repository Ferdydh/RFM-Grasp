import trimesh
import wandb
import torch
from einops import rearrange
from torch import vmap
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


def scene_to_wandb_image(scene: trimesh.Scene) -> wandb.Image:
    """
    Log a colored front view of a trimesh scene using PyVista's off-screen rendering.
    Returns a low-res wandb.Image for basic visualization.
    """
    import pyvista as pv
    import numpy as np

    # Convert trimesh scene to PyVista
    plotter = pv.Plotter(off_screen=True)

    # Add each mesh from the scene
    for geometry in scene.geometry.values():
        if hasattr(geometry, "vertices") and hasattr(geometry, "faces"):
            # Convert trimesh to PyVista
            mesh = pv.PolyData(
                geometry.vertices,
                np.hstack([[3] + face.tolist() for face in geometry.faces]),
            )

            # Handle color
            if hasattr(geometry, "visual") and hasattr(geometry.visual, "face_colors"):
                face_colors = geometry.visual.face_colors
                if face_colors is not None:
                    # Convert RGBA to RGB if needed
                    if face_colors.shape[1] == 4:
                        face_colors = face_colors[:, :3]
                    mesh.cell_data["colors"] = face_colors
                    plotter.add_mesh(mesh, scalars="colors", rgb=True)
            else:
                # Default color if no colors specified
                plotter.add_mesh(mesh, color="lightgray")

    # # Set a very low resolution
    plotter.window_size = [1024, 1024]

    # Get the image array
    img_array = plotter.screenshot(return_img=True)

    # Close the plotter
    plotter.close()

    return wandb.Image(img_array)


def rotmat_to_rotvec(matrix):
    """
    Convert rotation matrices to rotation vectors (axis-angle representation).
    This combines the previous quaternion conversion and vector conversion steps.

    Args:
        matrix: Batch of 3x3 rotation matrices
    Returns:
        Batch of 3D rotation vectors (axis-angle representation)
    """
    if len(matrix.shape) != 3 or matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    # Step 1: Convert rotation matrix to quaternion
    matrix = matrix.to(torch.float64)
    num_rots = matrix.shape[0]

    # Calculate diagonal and trace for quaternion conversion
    matrix_diag = torch.diagonal(matrix, dim1=-2, dim2=-1)
    matrix_trace = torch.sum(matrix_diag, dim=-1, keepdim=True)
    decision = torch.cat((matrix_diag, matrix_trace), dim=-1)
    choice = torch.argmax(decision, dim=-1)

    # Initialize quaternion output
    quat = torch.zeros((num_rots, 4), dtype=matrix.dtype, device=matrix.device)

    # Handle case where choice is not trace (not 3)
    not_three_mask = choice != 3
    i = choice[not_three_mask]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[not_three_mask, i] = (
        1 - decision[not_three_mask, 3] + 2 * matrix[not_three_mask, i, i]
    ).to(torch.float64)
    quat[not_three_mask, j] = (
        matrix[not_three_mask, j, i] + matrix[not_three_mask, i, j]
    ).to(torch.float64)
    quat[not_three_mask, k] = (
        matrix[not_three_mask, k, i] + matrix[not_three_mask, i, k]
    ).to(torch.float64)
    quat[not_three_mask, 3] = (
        matrix[not_three_mask, k, j] - matrix[not_three_mask, j, k]
    ).to(torch.float64)

    # Handle case where choice is trace (3)
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

    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    # Step 2: Convert quaternion to rotation vector
    quat = torch.where(quat[..., 3:4] < 0, -quat, quat)
    angle = 2.0 * torch.atan2(torch.norm(quat[..., :3], dim=-1), quat[..., 3])
    angle2 = angle * angle

    # Handle small and large angles differently
    small_scale = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
    large_scale = angle / torch.sin(angle / 2 + 1e-6)
    scale = torch.where(angle <= 1e-3, small_scale, large_scale)

    return scale[..., None] * quat[..., :3]


def sample_location_and_conditional_flow(x0, x1, t):
    """
    Compute conditional flow between two rotation matrices in SO(3) at specified time points.
    This implements a conditional flow matcher for the Special Orthogonal group in 3D.

    Args:
        x0: Starting rotation matrices (batch_size x 3 x 3)
        x1: Target rotation matrices (batch_size x 3 x 3)
        t: Time points between 0 and 1 (batch_size)

    Returns:
        xt: Interpolated rotation matrices at time t
        ut: Velocity field at time t (tangent vectors)
    """
    vec_manifold = SpecialOrthogonal(n=3, point_type="vector")

    # Convert rotations to axis-angle representation and compute log map
    rot_x0 = rotmat_to_rotvec(x0)
    rot_x1 = rotmat_to_rotvec(x1)
    log_x1 = vec_manifold.log_not_from_identity(rot_x1, rot_x0)

    # Ensure t requires gradient for velocity computation
    t.requires_grad = True

    # Compute interpolated rotation at time t
    xt = vec_manifold.exp_not_from_identity(t.reshape(-1, 1) * log_x1, rot_x0)
    xt = vec_manifold.matrix_from_rotation_vector(xt)

    # Compute velocity field using automatic differentiation
    xt_flat = rearrange(xt, "b c d -> b (c d)", c=3, d=3)

    def index_time_der(i):
        return torch.autograd.grad(xt_flat, t, i, create_graph=True, retain_graph=True)[
            0
        ]

    xt_dot = vmap(index_time_der, in_dims=1)(
        torch.eye(9).to(xt.device).repeat(xt_flat.shape[0], 1, 1)
    )
    ut = rearrange(xt_dot, "(c d) b -> b c d", c=3, d=3)

    return xt, ut
