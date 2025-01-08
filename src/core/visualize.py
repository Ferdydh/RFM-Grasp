from typing import List, Tuple, Union
import torch
import trimesh
import numpy as np
from scipy.spatial import cKDTree
import wandb

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
    mesh: Union[trimesh.Trimesh, trimesh.Scene], num_samples: int = 5000
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
    voxel_size: float = 0.001,  # Size of voxels for discretization
) -> Tuple[bool, float]:
    """Collision checking using point clouds and voxelization."""
    # Get points from both meshes
    gripper_points = _get_mesh_points(gripper_mesh)
    object_points = _get_mesh_points(object_mesh)

    # Print point clouds size for debugging
    # print(f"Number of gripper points: {len(gripper_points)}")
    # print(f"Number of object points: {len(object_points)}")

    # Discretize points to voxels
    gripper_voxels = np.round(gripper_points / voxel_size) * voxel_size
    object_voxels = np.round(object_points / voxel_size) * voxel_size

    # Remove duplicates after voxelization
    gripper_voxels = np.unique(gripper_voxels, axis=0)
    object_voxels = np.unique(object_voxels, axis=0)

    # First check: Axis-aligned bounding box intersection
    gripper_min = np.min(gripper_voxels, axis=0)
    gripper_max = np.max(gripper_voxels, axis=0)
    object_min = np.min(object_voxels, axis=0)
    object_max = np.max(object_voxels, axis=0)

    bbox_overlap = np.all(gripper_max >= object_min) and np.all(
        object_max >= gripper_min
    )
    if not bbox_overlap:
        return False, np.inf

    # Build KD-trees for efficient nearest neighbor search
    gripper_tree = cKDTree(gripper_voxels)
    object_tree = cKDTree(object_voxels)

    # Find minimum distances between point sets
    distances_g2o, _ = gripper_tree.query(object_voxels, k=1)
    distances_o2g, _ = object_tree.query(gripper_voxels, k=1)

    min_distance = min(np.min(distances_g2o), np.min(distances_o2g))
    # print(f"Minimum distance between point clouds: {min_distance}")

    # Check for overlapping points (considering discretization)
    collision_threshold = voxel_size * 1.5
    return min_distance < collision_threshold, min_distance


def check_collision(
    rotation_matrix: torch.Tensor,
    translation_vector: torch.Tensor,
    object_mesh_path: str,
    mesh_scale: float,
) -> Tuple[bool, trimesh.Scene, float]:
    """Checks for collisions between gripper and object."""
    # Load and scale object mesh
    object_mesh = trimesh.load(object_mesh_path)
    object_mesh.apply_scale(mesh_scale)

    object_mesh = enforce_trimesh(object_mesh)
    # object_mesh.vertices = object_mesh.vertices - object_mesh.centroid

    # Create transformation matrix
    gripper_transform = torch.eye(4)
    gripper_transform[:3, :3] = rotation_matrix[:3, :3]
    gripper_transform[:3, 3] = translation_vector.squeeze()
    gripper_transform = gripper_transform.numpy()

    # Create and transform gripper mesh
    gripper_mesh = create_parallel_gripper_mesh(color=[0, 255, 0])
    gripper_mesh.apply_transform(gripper_transform)

    # Print bounding boxes for debugging
    # print("Gripper bounds:", gripper_mesh.bounds)
    # if isinstance(object_mesh, trimesh.Scene):
    #     combined_bounds = np.array(
    #         [[np.inf, np.inf, np.inf], [-np.inf, -np.inf, -np.inf]]
    #     )
    #     for geom in object_mesh.geometry.values():
    #         if isinstance(geom, trimesh.Trimesh):
    #             combined_bounds[0] = np.minimum(combined_bounds[0], geom.bounds[0])
    #             combined_bounds[1] = np.maximum(combined_bounds[1], geom.bounds[1])
    #     print("Object bounds (combined):", combined_bounds)
    # else:
    #     print("Object bounds:", object_mesh.bounds)

    # Check collision
    has_collision, min_distance = _check_mesh_collision(gripper_mesh, object_mesh)

    # Update visualization color
    color = [255, 0, 0] if has_collision else [0, 255, 0]
    gripper_mesh.visual.face_colors = color

    # Create visualization
    if isinstance(object_mesh, trimesh.Scene):
        scene = object_mesh
        scene.add_geometry(gripper_mesh)
    else:
        scene = trimesh.Scene([object_mesh, gripper_mesh])

    return has_collision, scene, min_distance


def check_collision_multiple_grasps(
    rotation_matrix: torch.Tensor,
    translation_vector: torch.Tensor,
    object_mesh_path: str,
    mesh_scale: float,
) -> Tuple[bool, trimesh.Scene, float]:
    """Checks for collisions between multiple gripper poses and object.

    Args:
        rotation_matrix: Batch of rotation matrices (batch_size, 3, 3)
        translation_vector: Batch of translation vectors (batch_size, 3)
        object_mesh_path: Path to object mesh file
        mesh_scale: Scale factor for object mesh

    Returns:
        Tuple containing:
        - bool: True if any gripper has collision
        - trimesh.Scene: Scene with object and all gripper meshes
        - float: Minimum distance across all gripper-object pairs
    """
    # Load and scale object mesh
    object_mesh = trimesh.load(object_mesh_path)
    object_mesh.apply_scale(mesh_scale)
    object_mesh = enforce_trimesh(object_mesh)

    # Create transformation matrix
    batch_size = rotation_matrix.shape[0]
    gripper_meshes = []
    has_any_collision = False
    min_distance_overall = float("inf")

    for batch_idx in range(batch_size):
        # Extract single sample from batch
        so3_sample = rotation_matrix[batch_idx]
        r3_sample = translation_vector[batch_idx]

        gripper_transform = torch.eye(4)
        gripper_transform[:3, :3] = so3_sample[:3, :3]
        gripper_transform[:3, 3] = r3_sample.squeeze()
        gripper_transform = gripper_transform.numpy()

        # Create new gripper mesh for each sample
        gripper_mesh = create_parallel_gripper_mesh(color=[0, 255, 0])
        gripper_mesh.apply_transform(gripper_transform)

        # Check collision
        has_collision, min_distance = _check_mesh_collision(gripper_mesh, object_mesh)

        # Update overall collision status and minimum distance
        has_any_collision = has_any_collision or has_collision
        min_distance_overall = min(min_distance_overall, min_distance)

        # Update visualization color
        color = [255, 0, 0] if has_collision else [0, 255, 0]
        gripper_mesh.visual.face_colors = color

        # Store the gripper mesh
        gripper_meshes.append(gripper_mesh)

    # Create visualization with all gripper meshes
    if isinstance(object_mesh, trimesh.Scene):
        scene = object_mesh
        for gripper_mesh in gripper_meshes:
            scene.add_geometry(gripper_mesh)
    else:
        # Convert list of meshes to include object and all grippers
        all_meshes = [object_mesh] + gripper_meshes
        scene = trimesh.Scene(all_meshes)

    return has_any_collision, scene, min_distance_overall


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
