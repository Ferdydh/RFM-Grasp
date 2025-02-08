import os
import pickle
from collections import defaultdict
from pathlib import Path

import torch
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == "__main__":
    from scripts import initialize

    initialize()

    from src.core.config import ExperimentConfig
    from src.data.dataset import GraspDataset, MeshBatchSampler
    from src.models.flow import sample
    from src.models.lightning import Lightning
    from src.models.util import get_grasp_from_batch

    # Get all files in the grasps directory
    all_files = os.listdir("data/grasps")

    model = Lightning.load_from_checkpoint(
        "logs/checkpoints/run_20250202_233846/last.ckpt"
    )

    model.eval()

    config: ExperimentConfig = ExperimentConfig.default_mlp()
    config.data.sample_limit = None
    config.data.files = all_files

    # config.data.files = [
    #     "Pizza_caca4c8d409cddc66b04c0f74e5b376e_0.0065985560890656995.h5",
    # ]

    config.data.translation_norm_param_path = "logs/checkpoints/used_norm_params.pkl"

    # config.data.translation_norm_param_path = (
    #     "logs/checkpoints/run_20250204_204207/used_norm_params.pkl"
    # )

    config.data.dataset_workers = 8
    config.data.data_path = "data"

    data = GraspDataset(
        data_root=config.data.data_path,
        grasp_files=config.data.files,
        config=config,
        num_samples=None,
        device=model.device,
    )

    dl = DataLoader(
        dataset=data,
        # batch_sampler=MeshBatchSampler(data),
        shuffle=False,
        persistent_workers=True,
        num_workers=1,
        generator=torch.Generator(device=model.device),
        batch_size=1,
    )

    # Create output directory
    output_dir = Path("grasp_results")
    output_dir.mkdir(exist_ok=True)

    duplicate_list = []

    for batch in tqdm(dl, desc="Processing batches"):
        grasp_data = get_grasp_from_batch(batch)

        # Create a unique identifier using mesh path and normalization scale
        unique_id = (grasp_data.mesh_path, grasp_data.normalization_scale)

        # Skip if this combination already exists
        if unique_id in duplicate_list:
            # print(f"Skipping duplicate: {unique_id}")
            continue

        # Add to duplicate list
        duplicate_list.append(unique_id)

        sdf_input = rearrange(grasp_data.sdf, "... -> 1 1 ...")

        print("Sampling")

        so3_output, r3_output = sample(
            model.model,
            sdf_input,
            grasp_data.translation.device,
            torch.tensor(grasp_data.normalization_scale),
            num_samples=1024,
            sdf_path=grasp_data.mesh_path,
        )

        print("Saving")

        # Create output path by replacing .obj with .pkl and including normalization scale
        mesh_path = Path(grasp_data.mesh_path)
        output_path = (
            output_dir
            / f"{mesh_path.stem}_scale_{grasp_data.normalization_scale:.3f}.pkl"
        )

        # Prepare data to save
        batch_results = {"so3_output": so3_output.cpu(), "r3_output": r3_output.cpu()}

        # Save individual pickle file
        with open(output_path, "wb") as f:
            pickle.dump(batch_results, f)
