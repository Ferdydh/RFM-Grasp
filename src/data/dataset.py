from pytorch_lightning import LightningDataModule
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, dataset
from typing import Optional, List, Tuple, Union
import logging
import random
import json
from dataclasses import dataclass
from collections import namedtuple
from src.core.config import ExperimentConfig
from src.data.data_manager import GraspCache
from src.data.util import NormalizationParams, normalize_translation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# @dataclass
# class GraspData:
#     rotation: torch.Tensor
#     translation: torch.Tensor
#     sdf: torch.Tensor
#     mesh_path: str
#     #norm_params: NormalizationParams
#     dataset_mesh_scale: float
#     normalization_scale: float
#     centroid: np.ndarray
    
GraspData = namedtuple("GraspData", ["rotation", "translation", "sdf", "mesh_path", "dataset_mesh_scale", "normalization_scale", "centroid"])


# def collate_grasp_data(batch: list[GraspData]):
#     """
#     Custom collate function that takes a list of GraspData objects
#     and returns a dictionary of batched tensors and other fields.
#     """
#     rotations = []
#     translations = []
#     sdfs = []
#     mesh_paths = []
#     dataset_mesh_scales = []
#     normalization_scales = []
#     centroids = []

#     # Extract each field from the dataclass and accumulate in lists
#     for sample in batch:
#         rotations.append(sample.rotation)
#         translations.append(sample.translation)
#         sdfs.append(sample.sdf)
#         mesh_paths.append(sample.mesh_path)
#         dataset_mesh_scales.append(sample.dataset_mesh_scale)
#         normalization_scales.append(sample.normalization_scale)
#         centroids.append(sample.centroid)

#     # Convert lists of Tensors to a single batched Tensor
#     rotations = torch.stack(rotations)       # shape: (B, 3, 3)
#     translations = torch.stack(translations) # shape: (B, 3)
#     sdfs = torch.stack(sdfs)                 # shape: (B, 48, 48, 48), etc.

#     # For scales and centroids, either convert to Tensors or keep as lists:
#     dataset_mesh_scales = torch.tensor(dataset_mesh_scales)  # shape: (B,)
#     normalization_scales = torch.tensor(normalization_scales)# shape: (B,)
#     centroids = torch.stack([torch.tensor(c) for c in centroids])  # shape: (B, 3)

#     # Return a dictionary (or a new dataclass) with the collated batch
#     return GraspData(
#         "rotation": rotations,
#         "translation": translations,
#         "sdf": sdfs,
#         "mesh_path": mesh_paths,  # list of strings
#         "dataset_mesh_scale": dataset_mesh_scales,
#         "normalization_scale": normalization_scales,
#         "centroid": centroids,
#     )


class GraspDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        grasp_files: Union[List[str], int],
        split: str = "train",
        num_samples: Optional[int] = None,
        sdf_size: int = 48,
        device: torch.device = torch.device("cpu"),
    ):
        self.data_root = data_root
        self.sdf_size = sdf_size
        self.device = device
        self.cache = GraspCache(os.path.join(data_root, "grasp_cache"))

        # Process all grasp files
        self.grasp_entries = []
        total_grasps = 0
        # TODO: Why does normalization works better check it.
        if isinstance(grasp_files, int):
            # Perform globbing to get all .h5 files
            all_h5 = list(Path(self.data_root, "grasps").glob("*.h5"))
            random.seed(42)  # Fix the seed for reproducibility
            random.shuffle(all_h5)
            selected_files = all_h5[:grasp_files]
            # Extract only the filenames
            self.grasp_files = [f.name for f in selected_files]
        else:
            self.grasp_files = grasp_files
        
        for filename in self.grasp_files:
            entry = self.cache.get_or_process(filename, data_root, sdf_size)
            if  entry is None:
                continue
            num_grasps = len(entry.transforms)
            self.grasp_entries.append(
                (filename, total_grasps, total_grasps + num_grasps)
            )
            # Calculate min/max for normalization
            translations = entry.transforms[:, :3, 3] 
            if not hasattr(self, "trans_min"):
                self.trans_min = translations.min(axis=0)
                self.trans_max = translations.max(axis=0)
            else:
                self.trans_min = np.minimum(self.trans_min, translations.min(axis=0))
                self.trans_max = np.maximum(self.trans_max, translations.max(axis=0))
            total_grasps += num_grasps

        self.total_grasps = total_grasps

        self.norm_params = NormalizationParams(
            min=torch.tensor(self.trans_min), max=torch.tensor(self.trans_max)
        )
        print("Calculated min_max values: ", self.trans_min, self.trans_max)

        # Sample if needed
        if num_samples and num_samples < self.total_grasps:
            selected_indices = torch.randperm(self.total_grasps)[:num_samples]
            self.selected_indices = sorted(selected_indices.tolist())
            self.total_grasps = num_samples
        else:
            self.selected_indices = None

    def __len__(self):
        return self.total_grasps

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, str, NormalizationParams, float, float
    ]:
        if self.selected_indices is not None:
            idx = self.selected_indices[idx]

        # Find which grasp file contains this index
        entry_match = next(
            (entry for entry in self.grasp_entries if entry[1] <= idx < entry[2])
        )
        filename, start_idx, _ = entry_match
        entry = self.cache.cache[filename]
        grasp_idx = idx - start_idx

        rotation = torch.tensor(entry.transforms[grasp_idx][:3, :3], device=self.device)
        translation = torch.tensor(
            entry.transforms[grasp_idx][:3, 3], device=self.device
        )
        normalized_translation = normalize_translation(translation, self.norm_params)

        # return {
        #     'rotation':rotation,
        #     'translation':normalized_translation,
        #     'sdf':torch.tensor(entry.sdf),
        #     'mesh_path':entry.mesh_path,
        #     #norm_params:self.norm_params,
        #     'dataset_mesh_scale':entry.dataset_mesh_scale,
        #     'normalization_scale':entry.normalization_scale,
        #     'centroid':entry.centroid,
        # }
        return GraspData(
        rotation=rotation,
        translation=normalized_translation,
        sdf=torch.tensor(entry.sdf),
        mesh_path=entry.mesh_path,
        dataset_mesh_scale=entry.dataset_mesh_scale,
        normalization_scale=entry.normalization_scale,
        centroid=entry.centroid,
    )


class DataModule(LightningDataModule):
    def __init__(
        self,
        config: ExperimentConfig,
    ):
        super().__init__()
        self.config = config

        self.data_root = config.data.data_path
        self.grasp_files = config.data.files
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.num_samples = config.data.sample_limit
        self.split_ratio = config.data.split_ratio

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.split_ratio < 0.5:
            print("Warning: Split ratio is less than 0.5.")

    def setup(self, stage: Optional[str] = None):
        # Create split datasets
        if stage == "fit" or stage is None:
            # Create full dataset first
            self.full_dataset = GraspDataset(
                self.data_root,
                self.grasp_files,
                num_samples=self.num_samples,
                device=self.device,
            )
            self.save_used_files_to_json()
            # Calculate split sizes
            train_size = int(len(self.full_dataset) * self.split_ratio)
            val_size = len(self.full_dataset) - train_size

            if train_size == len(self.full_dataset) or val_size == 0 or train_size == 0:
                # This should only happen if we have sample_limit=1 or split_ratio=1.0
                print("Using the same dataset for training and validation.")
                self.train_dataset = self.full_dataset
                self.val_dataset = self.full_dataset
            else:
                # TODO: When it becomes a Subset it fails.
                self.train_dataset, self.val_dataset = dataset.random_split(
                    self.full_dataset,
                    [train_size, val_size],
                    generator=torch.Generator(device=self.device),
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            num_workers=self.num_workers,
            generator=torch.Generator(device=self.device),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=True,
            num_workers=self.num_workers,
            generator=torch.Generator(device=self.device),
        )
    
    def save_used_files_to_json(self):
        dirpath = self.config.training.checkpoint_dir + "/" + self.config.training.run_name
        os.makedirs(dirpath, exist_ok=True)  # Create the directory if it does not exist

        file_path = os.path.join(dirpath, "used_grasp_files.json")
        with open(file_path, 'w') as file:
            json.dump(self.full_dataset.grasp_files, file, indent=4)

