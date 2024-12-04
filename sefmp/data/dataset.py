import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Union, List, Tuple
from .data_manager import DataManager, DataSelector


class GraspDataset(Dataset):
    """Dataset for grasp learning, optimized for training."""

    def __init__(
        self,
        data_root: str,
        selectors: Union[DataSelector, List[DataSelector]],
        split: str = "train",
        num_samples: Optional[int] = None,
        sdf_size: int = 32,
        cache_dir: Optional[str] = None,
        compute_sdf: bool = True,
    ):
        self.data_manager = DataManager(
            data_root=data_root,
            selectors=selectors,
            cache_dir=cache_dir,
            sdf_size=sdf_size,
        )

        self.split = split
        self.num_samples = num_samples
        self.compute_sdf = compute_sdf

        # Create index of (mesh_path, transform_idx) pairs
        self.index = []
        for mesh_path in self.data_manager.get_all_mesh_paths():
            transforms = self.data_manager.get_transforms(mesh_path)
            num_transforms = len(transforms)
            self.index.extend([(mesh_path, i) for i in range(num_transforms)])

        # Optional: limit number of samples
        if self.num_samples and self.num_samples < len(self.index):
            indices = torch.randperm(len(self.index))[: self.num_samples]
            self.index = [self.index[i] for i in indices]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], float]:
        mesh_path, transform_idx = self.index[idx]

        # Get transform
        transform = self.data_manager.get_transforms(mesh_path)[transform_idx]
        rotation = transform[:3, :3]
        translation = transform[:3, 3]

        # Get SDF and scale if needed
        if self.compute_sdf:
            sdf, scale_factor, _ = self.data_manager.get_sdf(mesh_path)
            sdf_tensor = torch.tensor(sdf, dtype=torch.float32)
        else:
            sdf_tensor = None
            # Still need scale factor for transforms
            _, scale_factor, _ = self.data_manager.get_sdf(mesh_path)

        return rotation, translation, sdf_tensor, scale_factor


class GraspDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for grasp learning."""

    def __init__(
        self,
        data_root: str,
        selectors: Union[DataSelector, List[DataSelector]],
        batch_size: int = 32,
        num_workers: int = 4,
        num_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["selectors"])

        self.data_root = data_root
        self.selectors = selectors
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.cache_dir = cache_dir
        self.train_val_test_split = train_val_test_split

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = GraspDataset(
                self.data_root,
                self.selectors,
                split="train",
                num_samples=self.num_samples,
                cache_dir=self.cache_dir,
            )
            self.val_dataset = GraspDataset(
                self.data_root,
                self.selectors,
                split="val",
                num_samples=self.num_samples,
                cache_dir=self.cache_dir,
            )

        if stage == "test" or stage is None:
            self.test_dataset = GraspDataset(
                self.data_root,
                self.selectors,
                split="test",
                num_samples=self.num_samples,
                cache_dir=self.cache_dir,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
