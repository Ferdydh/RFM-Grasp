import os
import h5py
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataset
import trimesh
import mesh2sdf
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset
import pytorch_lightning as pl
import pickle
import logging
from pathlib import Path

from src.core.config import MLPExperimentConfig, TransformerExperimentConfig

# Set global default dtypes
torch.set_default_dtype(torch.float32)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Data structure for cached SDF and scale information."""

    sdf: np.ndarray
    normalization_scale: float
    timestamp: float


def enforce_trimesh(mesh) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Scene):
        return trimesh.util.concatenate(
            tuple(
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in mesh.geometry.values()
            )
        )
    elif isinstance(mesh, trimesh.Trimesh):
        return mesh
    else:
        raise ValueError(f"Unsupported mesh type: {type(mesh)}")


def process_mesh_to_sdf(
    mesh: trimesh.Trimesh, size: int = 32
) -> Tuple[np.ndarray, float]:
    """Process a mesh to SDF with consistent scaling and centering."""
    centroid = mesh.centroid
    vertices = mesh.vertices - centroid
    normalization_scale = np.float32(np.max(np.abs(vertices)))
    vertices = vertices / normalization_scale
    faces = mesh.faces

    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.uint32)

    # First compute raw SDF
    raw_sdf = mesh2sdf.compute(vertices, faces, size).astype(np.float32)
    abs_sdf = np.abs(raw_sdf)

    # Choose a level value within the range of the absolute SDF
    level = np.float32((abs_sdf.min() + abs_sdf.max()) / 2)

    # Compute final SDF with appropriate level
    sdf = mesh2sdf.compute(vertices, faces, size, fix=True, level=level).astype(
        np.float32
    )

    return sdf, normalization_scale


class SDFCache:
    """Handle caching of SDF computations."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "sdf_cache.pkl"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, CacheEntry]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Starting with empty cache.")
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get(self, mesh_path: str) -> Optional[Tuple[np.ndarray, float]]:
        if mesh_path not in self.cache:
            return None

        entry = self.cache[mesh_path]
        # Ensure cached values are float32
        sdf = entry.sdf.astype(np.float32)
        normalization_scale = np.float32(entry.normalization_scale)
        return sdf, normalization_scale

    def put(self, mesh_path: str, sdf: np.ndarray, normalization_scale: float):
        self.cache[mesh_path] = CacheEntry(
            sdf=sdf.astype(np.float32),
            normalization_scale=np.float32(normalization_scale),
            timestamp=os.path.getmtime(mesh_path),
        )
        self._save_cache()


class GraspDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        grasp_files: List[str],
        split: str = "train",
        num_samples: Optional[int] = None,
        sdf_size: int = 32,
        use_cache: bool = True,
    ):
        self.data_root = data_root
        self.grasp_files = grasp_files
        self.split = split
        self.num_samples = num_samples
        self.sdf_size = sdf_size
        self.use_cache = use_cache

        # Initialize cache
        self.cache = SDFCache(os.path.join(data_root, "sdf_cache"))

        # Process meshes and grasps
        self.mesh_to_sdf = {}
        self.mesh_to_normalization_scale = {}
        self.processed_data = []  # List to store all processed grasps

        # Process all grasp files and their corresponding meshes
        self._process_all_data()

        # Sample if needed
        if num_samples and num_samples < len(self.processed_data):
            indices = torch.randperm(len(self.processed_data))[:num_samples]
            self.processed_data = [self.processed_data[i] for i in indices]

    def _process_all_data(self):
        """Process all grasp files and their corresponding meshes."""
        total_files = len(self.grasp_files)

        for idx, grasp_filename in enumerate(self.grasp_files, 1):
            # Construct full path to grasp file
            grasp_file = os.path.join(self.data_root, "grasps", grasp_filename)
            logger.info(f"Processing grasp file {idx}/{total_files}: {grasp_filename}")

            # Load grasp file and get mesh information
            with h5py.File(grasp_file, "r") as h5file:
                mesh_fname = h5file["object/file"][()].decode("utf-8")
                dataset_mesh_scale = np.float32(h5file["object/scale"][()])

                # Load transforms and filter successful grasps
                transforms = h5file["grasps"]["transforms"][:].astype(np.float32)
                grasp_success = h5file["grasps"]["qualities"]["flex"][
                    "object_in_gripper"
                ][:]
                transforms = transforms[grasp_success == 1]

            # Load and scale mesh
            mesh_path = os.path.join(self.data_root, mesh_fname)

            # Process mesh to SDF if not in cache
            if mesh_path not in self.mesh_to_sdf:
                cache_result = self.cache.get(mesh_path)

                if self.use_cache and cache_result is not None:
                    sdf, normalization_scale = cache_result
                    logger.info(f"Loaded {os.path.basename(mesh_path)} from cache")
                else:
                    mesh = trimesh.load(mesh_path)
                    mesh = mesh.apply_scale(dataset_mesh_scale)
                    mesh = enforce_trimesh(mesh)

                    logger.info(f"Computing SDF for {os.path.basename(mesh_path)}")
                    sdf, normalization_scale = process_mesh_to_sdf(mesh, self.sdf_size)
                    self.cache.put(mesh_path, sdf, normalization_scale)

                self.mesh_to_sdf[mesh_path] = sdf
                self.mesh_to_normalization_scale[mesh_path] = normalization_scale

            # Scale and store transforms
            normalization_scale = self.mesh_to_normalization_scale[mesh_path]
            for transform in transforms:
                scaled_transform = transform.copy() / normalization_scale
                self.processed_data.append(
                    {
                        "mesh_path": mesh_path,
                        "transform": torch.tensor(
                            scaled_transform, dtype=torch.float32
                        ),
                        "dataset_mesh_scale": np.float32(dataset_mesh_scale),
                        "normalization_scale": np.float32(normalization_scale),
                    }
                )

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, float, float]:
        data_point = self.processed_data[idx]
        mesh_path = data_point["mesh_path"]
        transform = data_point["transform"]
        dataset_mesh_scale = data_point["dataset_mesh_scale"]
        normalization_scale = data_point["normalization_scale"]

        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        sdf = torch.tensor(self.mesh_to_sdf[mesh_path], dtype=torch.float32)

        return (
            rotation,
            translation,
            sdf,
            mesh_path,
            dataset_mesh_scale,
            normalization_scale,
        )


class DataHandler(pl.LightningDataModule):
    def __init__(
        self,
        config: MLPExperimentConfig | TransformerExperimentConfig,
    ):
        super().__init__()
        self.config = config

        self.data_root = config.data.data_path
        self.grasp_files = config.data.files
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.num_samples = config.data.sample_limit
        self.split_ratio = config.data.split_ratio

        # Important: Set num_workers=0 when using MPS
        if torch.backends.mps.is_available():
            self.num_workers = 0

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

            print("train_size", train_size)
            print("val_size", val_size)
            print("len(full_dataset)", len(full_dataset))

            if train_size == len(full_dataset) or val_size == 0 or train_size == 0:
                # This should only happen if we have sample_limit=1 or split_ratio=1.0
                self.train_dataset = full_dataset
                self.val_dataset = full_dataset
            else:
                self.train_dataset, self.val_dataset = (
                    torch.utils.data.dataset.random_split(
                        full_dataset, [train_size, val_size]
                    )
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
