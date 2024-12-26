from pytorch_lightning import LightningDataModule
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, dataset
from typing import Optional, List, Tuple
import logging
from collections import namedtuple

from src.core.config import BaseExperimentConfig
from src.data.data_manager import GraspCache
from src.data.util import NormalizationParams, normalize_translation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraspDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        grasp_files: List[str],
        split: str = "train",
        num_samples: Optional[int] = None,
        sdf_size: int = 32,
    ):
        self.data_root = data_root
        self.sdf_size = sdf_size
        self.cache = GraspCache(os.path.join(data_root, "grasp_cache"))

        # Process all grasp files
        self.grasp_entries = []
        total_grasps = 0

        for filename in grasp_files:
            entry = self.cache.get_or_process(filename, data_root, sdf_size)
            num_grasps = len(entry.transforms)
            self.grasp_entries.append(
                (filename, total_grasps, total_grasps + num_grasps)
            )
                        # Calculate min/max for normalization
            translations = entry.transforms[:, :3, 3]  # Get translation parts
            if not hasattr(self, 'trans_min'):
                self.trans_min = translations.min(axis=0)
                self.trans_max = translations.max(axis=0)
            else:
                self.trans_min = np.minimum(self.trans_min, translations.min(axis=0))
                self.trans_max = np.maximum(self.trans_max, translations.max(axis=0))
            total_grasps += num_grasps

        self.total_grasps = total_grasps

        self.norm_params = NormalizationParams(
            min=torch.tensor(self.trans_min),
            max=torch.tensor(self.trans_max)
        )
        print("Calculated min_max values: ",self.trans_min,self.trans_max)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, float, float, NormalizationParams]:
        if self.selected_indices is not None:
            idx = self.selected_indices[idx]

        # Find which grasp file contains this index
        entry_match = next(
            (entry for entry in self.grasp_entries if entry[1] <= idx < entry[2])
        )
        filename, start_idx, _ = entry_match
        entry = self.cache.cache[filename]
        grasp_idx = idx - start_idx

        rotation = torch.tensor(entry.transforms[grasp_idx][:3, :3])
        translation = torch.tensor(entry.transforms[grasp_idx][:3, 3])
        normalized_translation = normalize_translation(translation, self.norm_params)


        return (
            rotation,
            normalized_translation,
            self.norm_params,
            torch.tensor(entry.sdf),
            entry.mesh_path,
            entry.dataset_mesh_scale,
            entry.normalization_scale,
        )


class DataModule(LightningDataModule):
    def __init__(
        self,
        config: BaseExperimentConfig,
    ):
        super().__init__()
        self.config = config

        self.data_root = config.data.data_path
        self.grasp_files = config.data.files
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.num_samples = config.data.sample_limit
        self.split_ratio = config.data.split_ratio

        if self.split_ratio < 0.5:
            print("Warning: Split ratio is less than 0.5.")

    def setup(self, stage: Optional[str] = None):
        # Create split datasets
        if stage == "fit" or stage is None:
            # Create full dataset first
            full_dataset = GraspDataset(
                self.data_root,
                self.grasp_files,
                num_samples=self.num_samples,
            )

            # Calculate split sizes
            train_size = int(len(full_dataset) * self.split_ratio)
            val_size = len(full_dataset) - train_size

            if train_size == len(full_dataset) or val_size == 0 or train_size == 0:
                # This should only happen if we have sample_limit=1 or split_ratio=1.0
                self.train_dataset = full_dataset
                self.val_dataset = full_dataset
            else:
                self.train_dataset, self.val_dataset = dataset.random_split(
                    full_dataset, [train_size, val_size]
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=True,
            num_workers=self.num_workers,
        )
