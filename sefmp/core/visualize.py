import torch
import trimesh
import numpy as np

from core.utils import load_config
from data.grasp_dataset import GraspDataset


def create_gripper_marker(color=[0, 0, 255], sections=6) -> trimesh.Trimesh:
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp


def check_collision(gripper_mesh: trimesh.Trimesh, obj_mesh: trimesh.Trimesh) -> bool:
    """Check if there is a collision between the gripper and object meshes.

    Args:
        gripper_mesh (trimesh.Trimesh): The gripper mesh
        obj_mesh (trimesh.Trimesh): The object mesh

    Returns:
        bool: True if there is a collision, False otherwise
    """
    # Check for collisions between the meshes
    # We use the built-in collision manager from trimesh
    manager = trimesh.collision.CollisionManager()
    manager.add_object("object", obj_mesh)

    # Check if the gripper collides with the object
    is_collision = manager.in_collision_single(gripper_mesh)

    return is_collision


def visualize(experiment: str = "visualize"):
    """Visualize an existing grasp from the dataset and check for collisions."""

    cfg = load_config(experiment)

    test = GraspDataset(
        data_root=cfg["data"]["data_path"],
        grasp_files=cfg["data"]["grasp_files"],
        num_samples=cfg["data"]["num_samples"],
        split="test",
        use_cache=False,
    )

    (
        so3_input,
        r3_input,
        sdf_input,
        mesh_path,
        dataset_mesh_scale,
        normalization_scale,
    ) = test[0]

    print("SO3 Input:", so3_input)
    print("R3 Input:", r3_input)
    print("Mesh Path:", mesh_path)
    print("Normalization Scale:", normalization_scale)
    print("Dataset Mesh Scale:", dataset_mesh_scale)

    # Load and scale the object mesh
    obj_mesh = trimesh.load(mesh_path)
    obj_mesh = obj_mesh.apply_scale(dataset_mesh_scale)

    # Create the transformation matrix
    transform = torch.eye(4)
    transform[:3, :3] = so3_input[:3, :3] * normalization_scale
    transform[:3, 3] = r3_input.squeeze() * normalization_scale
    transform = transform.numpy()  # Convert to numpy for trimesh

    print("Transform:", transform)

    # Create the gripper mesh with initial green color
    gripper = create_gripper_marker(color=[0, 255, 0])
    # Apply the transform to position the gripper
    gripper = gripper.apply_transform(transform)

    # Check for collisions
    has_collision = check_collision(gripper, obj_mesh)

    # Set the color based on collision status
    color = (
        [255, 0, 0] if has_collision else [0, 255, 0]
    )  # Red if collision, green if not
    gripper.visual.face_colors = color

    # Create the visualization scene
    scene = trimesh.Scene([obj_mesh, gripper])

    # Print collision status
    print(f"Collision detected: {has_collision}")

    # Show the scene
    scene.show()
