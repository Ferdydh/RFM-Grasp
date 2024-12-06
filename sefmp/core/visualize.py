import torch
import trimesh

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


def visualize(experiment: str = "visualize"):
    """Visualize an existing grasp from the dataset."""

    cfg = load_config(experiment)

    test = GraspDataset(
        data_root=cfg["data"]["data_path"],
        grasp_files=cfg["data"]["grasp_files"],  # Using list of selectors
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
    # print("SDF Input:", sdf_input)
    print("Mesh Path:", mesh_path)
    print("Normalization Scale:", normalization_scale)
    print("Dataset Mesh Scale:", dataset_mesh_scale)

    obj_mesh = trimesh.load(mesh_path)
    obj_mesh = obj_mesh.apply_scale(dataset_mesh_scale)

    transform = torch.eye(4)
    transform[:3, :3] = so3_input[:3, :3] * normalization_scale
    transform[:3, 3] = r3_input.squeeze() * normalization_scale
    transform = transform

    print("Transform:", transform)

    # create visual markers for grasps
    successful_grasps = [
        create_gripper_marker(color=[0, 255, 0]).apply_transform(transform)
    ]

    trimesh.Scene([obj_mesh] + successful_grasps).show()
