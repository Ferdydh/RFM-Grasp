from typing import List, Tuple, Union

import numpy as np
import torch
import trimesh
import wandb
from scipy.spatial import cKDTree

from src.data.util import enforce_trimesh


def create_parallel_gripper_mesh(
    color: List[int] = [0, 0, 255],
    cylinder_sections: int = 12,
    gripper_width: float = 0.082,
    gripper_height: float = 0.11217,
    finger_thickness: float = 0.002,
    base_height: float = 0.066,
) -> trimesh.Trimesh:
    """Creates a 3D mesh representing a parallel-jaw gripper."""
    half_width = gripper_width / 2

    # Create basic components
    left_finger = trimesh.creation.cylinder(
        radius=finger_thickness,
        sections=cylinder_sections,
        segment=[
            [half_width, 0, base_height],
            [half_width, 0, gripper_height],
        ],
    )

    right_finger = trimesh.creation.cylinder(
        radius=finger_thickness,
        sections=cylinder_sections,
        segment=[
            [-half_width, 0, base_height],
            [-half_width, 0, gripper_height],
        ],
    )

    base_cylinder = trimesh.creation.cylinder(
        radius=finger_thickness,
        sections=cylinder_sections,
        segment=[[0, 0, 0], [0, 0, base_height]],
    )

    connector = trimesh.creation.cylinder(
        radius=finger_thickness,
        sections=cylinder_sections,
        segment=[[-half_width, 0, base_height], [half_width, 0, base_height]],
    )

    # Combine components
    gripper_mesh = trimesh.util.concatenate(
        [base_cylinder, connector, right_finger, left_finger]
    )
    gripper_mesh.visual.face_colors = color

    return gripper_mesh


def _get_mesh_points(
    mesh: Union[trimesh.Trimesh, trimesh.Scene], num_samples: int = 10000
) -> np.ndarray:
    """Extract points from a mesh or scene, handling both cases appropriately."""
    if isinstance(mesh, trimesh.Scene):
        # Combine points from all geometries in the scene
        points = []
        for geom in mesh.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                # Sample points from the surface
                sampled = geom.sample(num_samples // len(mesh.geometry))
                points.append(sampled)
                # Add vertices
                points.append(geom.vertices)
        return np.vstack(points)
    else:
        # For single mesh, combine sampled points and vertices
        sampled = mesh.sample(num_samples)
        return np.vstack([sampled, mesh.vertices])


def _check_mesh_collision(
    gripper_mesh: trimesh.Trimesh,
    object_mesh: Union[trimesh.Trimesh, trimesh.Scene],
    voxel_size: float = 0.0003,  # Balanced voxel size
    collision_margin: float = 0.0004,  # Balanced collision margin
) -> Tuple[bool, float]:
    """Collision checking using point clouds and voxelization with improved accuracy."""
    # Get points from both meshes with increased sampling
    gripper_points = _get_mesh_points(gripper_mesh, num_samples=10000)
    object_points = _get_mesh_points(object_mesh, num_samples=10000)

    # Perform initial AABB check with small margin
    gripper_min = np.min(gripper_points, axis=0)
    gripper_max = np.max(gripper_points, axis=0)
    object_min = np.min(object_points, axis=0)
    object_max = np.max(object_points, axis=0)

    margin = voxel_size * 2
    bbox_overlap = np.all(gripper_max + margin >= object_min) and np.all(
        object_max + margin >= gripper_min
    )

    if not bbox_overlap:
        return False, np.inf

    # Discretize points to voxels
    gripper_voxels = np.round(gripper_points / voxel_size) * voxel_size
    object_voxels = np.round(object_points / voxel_size) * voxel_size

    # Remove duplicates after voxelization
    gripper_voxels = np.unique(gripper_voxels, axis=0)
    object_voxels = np.unique(object_voxels, axis=0)

    # Build KD-trees for efficient nearest neighbor search
    gripper_tree = cKDTree(gripper_voxels)
    object_tree = cKDTree(object_voxels)

    # First check: Use trimesh's built-in collision detection if available
    try:
        if not isinstance(object_mesh, trimesh.Scene):
            # Check exact intersection first
            mesh_collision = trimesh.collision.collides.mesh_intersects(
                gripper_mesh, object_mesh
            )
            if mesh_collision:
                return True, 0.0
    except:
        pass  # Fall back to point-based method if trimesh collision fails

    # Find minimum distances between point sets
    k = 3  # Balanced number of nearest neighbors
    distances_g2o, _ = gripper_tree.query(object_voxels, k=k)
    distances_o2g, _ = object_tree.query(gripper_voxels, k=k)

    # Use minimum distance for each point (more sensitive to actual collisions)
    min_distance_g2o = np.min(distances_g2o[:, 0])  # Only look at closest neighbor
    min_distance_o2g = np.min(distances_o2g[:, 0])  # Only look at closest neighbor

    min_distance = min(min_distance_g2o, min_distance_o2g)

    # Count points that are very close
    close_points_count = np.sum(distances_g2o[:, 0] < collision_margin) + np.sum(
        distances_o2g[:, 0] < collision_margin
    )

    # Detect collision if either:
    # 1. Points are extremely close (definite collision)
    # 2. Multiple points are within collision margin (probable collision)
    has_collision = (min_distance < collision_margin / 2) or (
        min_distance < collision_margin and close_points_count >= 3
    )

    return has_collision, min_distance


def check_collision(
    rotation_matrix: torch.Tensor,
    translation_vector: torch.Tensor,
    object_mesh_path: str,
    mesh_scale: float,
) -> Tuple[bool, trimesh.Scene, float]:
    """Checks for collisions between gripper poses and object with improved accuracy."""
    # Load and scale object mesh
    object_mesh = trimesh.load(object_mesh_path)

    if torch.is_tensor(mesh_scale):
        mesh_scale = mesh_scale.cpu().numpy()
    object_mesh.apply_scale(mesh_scale)
    object_mesh = enforce_trimesh(object_mesh)

    # Check if rotation matrix is SO3 (3x3) or batched (Nx3x3)
    is_so3 = rotation_matrix.shape == torch.Size([3, 3])
    if is_so3:
        rotation_matrix = rotation_matrix.unsqueeze(0)
        translation_vector = translation_vector.unsqueeze(0)

    # Validate shapes
    assert len(rotation_matrix.shape) == 3 and rotation_matrix.shape[1:] == (3, 3), (
        f"Expected rotation matrix shape (N, 3, 3), got {rotation_matrix.shape}"
    )
    assert len(translation_vector.shape) == 2 and translation_vector.shape[1] == 3, (
        f"Expected translation vector shape (N, 3), got {translation_vector.shape}"
    )
    assert rotation_matrix.shape[0] == translation_vector.shape[0], (
        f"Batch sizes don't match: {rotation_matrix.shape[0]} != {translation_vector.shape[0]}"
    )

    batch_size = rotation_matrix.shape[0]
    gripper_meshes = []
    has_any_collision = False
    min_distance_overall = float("inf")

    # Process each grasp
    for batch_idx in range(batch_size):
        # Create transformation matrix
        gripper_transform = torch.eye(4)
        gripper_transform[:3, :3] = rotation_matrix[batch_idx]
        gripper_transform[:3, 3] = translation_vector[batch_idx]
        gripper_transform = gripper_transform.cpu().numpy()

        # Create and transform gripper mesh
        gripper_mesh = create_parallel_gripper_mesh(color=[0, 255, 0])
        gripper_mesh.apply_transform(gripper_transform)

        # Check collision with balanced parameters
        has_collision, min_distance = _check_mesh_collision(
            gripper_mesh,
            object_mesh,
            voxel_size=0.0003,  # Balanced voxel size
            collision_margin=0.0004,  # Balanced collision margin
        )

        # Update overall collision status and minimum distance
        has_any_collision = has_any_collision or has_collision
        min_distance_overall = min(min_distance_overall, min_distance)

        # Update visualization color
        color = [255, 0, 0] if has_collision else [0, 255, 0]
        gripper_mesh.visual.face_colors = color

        gripper_meshes.append(gripper_mesh)

    # Create visualization
    if isinstance(object_mesh, trimesh.Scene):
        scene = object_mesh
        for gripper_mesh in gripper_meshes:
            scene.add_geometry(gripper_mesh)
    else:
        all_meshes = [object_mesh] + gripper_meshes
        scene = trimesh.Scene(all_meshes)

    return has_any_collision, scene, min_distance_overall


def scene_to_wandb_3d(scene: trimesh.Scene) -> wandb.Object3D:
    """Convert trimesh scene to wandb 3D object."""
    scene.export("logs/mesh.glb")
    return wandb.Object3D("logs/mesh.glb")
