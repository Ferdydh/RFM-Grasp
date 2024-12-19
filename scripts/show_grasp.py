from src.core.visualize import check_collision
from src.data.grasp_dataset import GraspDataset
from src.core.config import DataConfig
from src.models.util import scene_to_wandb_image


if __name__ == "__main__":
    config = DataConfig.sanity()

    test = GraspDataset(
        data_root=config.data_path,
        grasp_files=config.files,
        num_samples=config.sample_limit,
        split="test",
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
    print("Dataset Mesh Scale:", dataset_mesh_scale)

    has_collision, scene, min_distance = check_collision(
        so3_input,
        r3_input,
        mesh_path,
        dataset_mesh_scale,
    )

    # Print collision status
    print(f"Collision: {has_collision}")
    print(f"Minimum Distance: {min_distance}")

    # Show the scene
    scene.show()

    img = scene_to_wandb_image(scene)
    img.image.show()
