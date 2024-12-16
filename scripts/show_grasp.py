from src.core.visualize import check_collision
from src.data.grasp_dataset import GraspDataset
from src.core.config import DataConfig


if __name__ == "__main__":
    config = DataConfig.sanity()

    test = GraspDataset(
        data_root=config.data_path,
        grasp_files=config.files,
        num_samples=config.sample_limit,
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

    r3_input[2] = r3_input[2] + 0.0002

    has_collision, scene, min_distance = check_collision(
        so3_input, r3_input, mesh_path, dataset_mesh_scale, normalization_scale
    )

    # Print collision status
    print(f"Collision: {has_collision}")
    print(f"Minimum Distance: {min_distance}")

    # Show the scene
    scene.show()
