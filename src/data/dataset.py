from pytorch_lightning import LightningDataModule
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, dataset
from typing import Optional, List, Tuple, Union
import logging
import random

from src.core.config import ExperimentConfig
from src.data.data_manager import GraspCache
from src.data.util import NormalizationParams, normalize_translation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            grasp_files = [f.name for f in selected_files]
        else:
            grasp_files = grasp_files
        
        #print("Grasp files: ", grasp_files)
        for filename in grasp_files:
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
        torch.Tensor, torch.Tensor, torch.Tensor, str, float, float, NormalizationParams
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

        return (
            rotation,
            normalized_translation,
            torch.tensor(entry.sdf),
            entry.mesh_path,
            self.norm_params,
            entry.dataset_mesh_scale,
            entry.normalization_scale,
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
            full_dataset = GraspDataset(
                self.data_root,
                self.grasp_files,
                num_samples=self.num_samples,
                device=self.device,
            )

            # Calculate split sizes
            train_size = int(len(full_dataset) * self.split_ratio)
            val_size = len(full_dataset) - train_size

            if train_size == len(full_dataset) or val_size == 0 or train_size == 0:
                # This should only happen if we have sample_limit=1 or split_ratio=1.0
                print("Using the same dataset for training and validation.")
                self.train_dataset = full_dataset
                self.val_dataset = full_dataset
            else:
                # TODO: When it becomes a Subset it fails.
                self.train_dataset, self.val_dataset = dataset.random_split(
                    full_dataset,
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
