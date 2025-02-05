import torch

from src.data.util import denormalize_translation

if __name__ == "__main__":
    from scripts import initialize

    initialize()

    from src.core.config import ExperimentConfig
    from src.core.visualize import check_collision
    from src.data.dataset import GraspDataset
    from src.data.util import GraspData

    config = ExperimentConfig.default_mlp()
    config.data.translation_norm_param_path = "logs/checkpoints/used_norm_params.pkl"

    test = GraspDataset(
        data_root=config.data.data_path,
        config=config,
        grasp_files=config.data.files,
        num_samples=config.data.sample_limit,
        split="test",
    )

    grasp_data: GraspData = test[0]

    # Denormalize and adjust translation with centroid
    denormalized_translation = denormalize_translation(
        grasp_data.translation, test.norm_params
    )
    final_translation = denormalized_translation + torch.tensor(
        grasp_data.centroid, device=denormalized_translation.device
    )

    move = torch.tensor([0.011, 0, 0])
    # move = torch.tensor([0.014, 0, 0])
    # move = torch.tensor([0.0113, 0, 0])
    final_translation = final_translation + move

    has_collision, scene, min_distance = check_collision(
        grasp_data.rotation,
        final_translation,
        grasp_data.mesh_path,
        grasp_data.dataset_mesh_scale,
    )

    scene.show()

    # scene.export("logs/output.glb")
