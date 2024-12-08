import os
import glob
import h5py
import torch
import trimesh
import mesh2sdf
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict
from torch.utils.data import Dataset, DataLoader,Sampler,RandomSampler
import pytorch_lightning as pl
from collections import defaultdict
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSelector:
    """
    Flexible data selection criteria for grasps dataset.
    """

    grasp_id: Optional[str] = None
    object_id: Optional[str] = None
    item_name: Optional[str] = None

    def matches(self, mesh_path: str, grasp_path: str) -> bool:
        """Check if paths match the selection criteria.
        We don't use the mesh_path here, because we know the mesh exists.
        """
        grasp_filename = os.path.basename(grasp_path)
        if "_" in grasp_filename:
            item_name, obj_id, grasp_value = grasp_filename.split("_")
            grasp_value = grasp_value.replace(".h5", "")
        else:
            return False

        if self.grasp_id and self.grasp_id != grasp_value:
            return False
        if self.object_id and self.object_id != obj_id:
            return False
        if self.item_name and self.item_name != item_name:
            return False
        return True


@dataclass
class CacheEntry:
    """Data structure for cached SDF and scale information."""

    sdf: np.ndarray
    scale_factor: float
    timestamp: float  # For tracking cache freshness
    mesh_hash: str  # For validating mesh contents


def compute_mesh_hash(mesh_path: str) -> str:
    """Compute a hash of the mesh file contents."""
    with open(mesh_path, "rb") as f:
        return str(hash(f.read()))


def process_mesh_to_sdf(mesh_path: str, size: int = 32) -> Tuple[np.ndarray, float]:
    """Process a mesh to SDF with consistent scaling and centering."""
    mesh = trimesh.load(mesh_path)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in mesh.geometry.values()
            )
        )

    centroid = mesh.centroid
    vertices = mesh.vertices - centroid
    scale_factor = np.max(np.abs(vertices))
    vertices = vertices / scale_factor
    faces = mesh.faces

    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.uint32)

    # First compute raw SDF
    raw_sdf = mesh2sdf.core.compute(vertices, faces, size)
    abs_sdf = np.abs(raw_sdf)

    # Choose a level value within the range of the absolute SDF
    level = (abs_sdf.min() + abs_sdf.max()) / 2

    # Compute final SDF with appropriate level
    sdf = mesh2sdf.compute(vertices, faces, size, fix=True, level=level)

    return sdf, scale_factor


class SDFCache:
    """Handle caching of SDF computations."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "sdf_cache.pkl"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, CacheEntry]:
        """Load cache from disk if it exists."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Starting with empty cache.")
                return {}
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get(self, mesh_path: str, size: int) -> Optional[Tuple[np.ndarray, float]]:
        """Get SDF and scale factor from cache if available and valid."""
        if mesh_path not in self.cache:
            return None

        entry = self.cache[mesh_path]
        current_hash = compute_mesh_hash(mesh_path)

        if current_hash != entry.mesh_hash:
            logger.info(f"Mesh {mesh_path} has changed, recomputing SDF")
            return None

        return entry.sdf, entry.scale_factor

    def put(self, mesh_path: str, sdf: np.ndarray, scale_factor: float):
        """Store SDF and scale factor in cache."""
        self.cache[mesh_path] = CacheEntry(
            sdf=sdf,
            scale_factor=scale_factor,
            timestamp=os.path.getmtime(mesh_path),
            mesh_hash=compute_mesh_hash(mesh_path),
        )
        self._save_cache()

class RepeatSampler(Sampler):
    """Required for creating batches larger than the dataset."""

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_repetitions = (batch_size + len(data_source) - 1) // len(data_source)
    def __iter__(self):
        # Repeat the indices to fill the batch
        indices = list(range(len(self.data_source))) * self.num_repetitions
        return iter(indices[:self.batch_size])

    def __len__(self):
        return self.batch_size
    


class GraspDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        selectors: Union[DataSelector, List[DataSelector]],
        split: str = "train",
        num_samples: Optional[int] = None,
        sdf_size: int = 32,
        cache_dir: Optional[str] = None,
    ):
        self.data_root = data_root
        self.selectors = (
            [selectors] if isinstance(selectors, DataSelector) else selectors
        )
        self.split = split
        self.num_samples = num_samples
        self.sdf_size = sdf_size

        # Initialize cache
        self.cache = SDFCache(cache_dir or os.path.join(data_root, "sdf_cache"))

        # Get all meshes and grasps
        mesh_pattern = os.path.join(data_root, "meshes/**/*.obj")
        grasp_pattern = os.path.join(data_root, "grasps/*.h5")

        self.meshes = glob.glob(mesh_pattern, recursive=True)
        self.grasps = glob.glob(grasp_pattern)

        # Group grasps by mesh
        self.mesh_to_grasps = self._create_mesh_grasp_mapping()

        # Process meshes to SDFs using cache
        self.mesh_to_sdf = {}
        self.mesh_to_scale = {}
        self._compute_all_sdfs()

        # Load transforms
        self.mesh_to_transforms = {}
        self._load_all_transforms()

        # Create index
        self._create_index()

    def _compute_all_sdfs(self):
        """Compute or load from cache SDFs for all unique meshes."""
        total_meshes = len(self.mesh_to_grasps)
        for idx, mesh_path in enumerate(self.mesh_to_grasps.keys(), 1):
            logger.info(
                f"Processing mesh {idx}/{total_meshes}: {os.path.basename(mesh_path)}"
            )

            # Try to get from cache
            cache_result = self.cache.get(mesh_path, self.sdf_size)

            if cache_result is not None:
                sdf, scale_factor = cache_result
                logger.info(f"Loaded {os.path.basename(mesh_path)} from cache")
            else:
                logger.info(f"Computing SDF for {os.path.basename(mesh_path)}")
                sdf, scale_factor = process_mesh_to_sdf(mesh_path, self.sdf_size)
                self.cache.put(mesh_path, sdf, scale_factor)

            self.mesh_to_sdf[mesh_path] = sdf
            self.mesh_to_scale[mesh_path] = scale_factor

    def _load_all_transforms(self):
        """Load and scale transforms for all meshes."""
        for mesh_path, grasp_files in self.mesh_to_grasps.items():
            scale_factor = self.mesh_to_scale[mesh_path]

            transforms_list = []
            for grasp_file in grasp_files:
                with h5py.File(grasp_file, "r") as h5file:
                    transforms = h5file["grasps"]["transforms"][:]
                    grasp_sucess = h5file['grasps']['qualities']['flex']['object_in_gripper'][:]
                    mask = grasp_sucess == 1
                    transforms = transforms[mask]
                    # Scale translations
                    for transform in transforms:
                        #TODO: Correct this
                        transform[:3, 3] = transform[:3, 3] #/ scale_factor * 2.0
                    transforms_list.extend(transforms)

            self.mesh_to_transforms[mesh_path] = torch.tensor(
                transforms_list, dtype=torch.float32
            )

    def _create_mesh_grasp_mapping(self) -> Dict[str, List[str]]:
        """Create mapping from mesh paths to corresponding grasp files."""
        mesh_to_grasps = defaultdict(list)

        for grasp in self.grasps:
            grasp_filename = os.path.basename(grasp)
            if "_" not in grasp_filename:
                continue

            item_name, obj_id, _ = grasp_filename.split("_", 2)
            matching_meshes = [m for m in self.meshes if obj_id in m and item_name in m]

            if not matching_meshes:
                continue

            mesh = matching_meshes[0]
            if any(selector.matches(mesh, grasp) for selector in self.selectors):
                mesh_to_grasps[mesh].append(grasp)

        return mesh_to_grasps

    def _create_index(self):
        """Create flat index for accessing data."""
        self.index = []
        for mesh_path, transforms in self.mesh_to_transforms.items():
            num_transforms = len(transforms)
            self.index.extend([(mesh_path, i) for i in range(num_transforms)])

        if self.num_samples and self.num_samples < len(self.index):
            indices = torch.randperm(len(self.index))[: self.num_samples]
            self.index = [self.index[i] for i in indices]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mesh_path, transform_idx = self.index[idx]

        transform = self.mesh_to_transforms[mesh_path][transform_idx]
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        sdf = torch.tensor(self.mesh_to_sdf[mesh_path])
        return rotation, translation, sdf


class GraspDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        selectors: Union[DataSelector, List[DataSelector]],
        sampler_opt: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        num_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        super().__init__()
        self.data_root = data_root
        self.selectors = selectors
        self.sampler_opt = sampler_opt # to create batches bigger than the dataset
        self.batch_size = batch_size
        self.num_workers = num_workers # currently it is 0 because new workers are created after each epoch
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
        num_samples = self.batch_size if self.sampler_opt == 'repeat' else len(self.train_dataset)
        self.sampler = RandomSampler(data_source = self.train_dataset,num_samples = num_samples)

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
            #shuffle=True,
            num_workers=self.num_workers,
            sampler = self.sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=8,#self.batch_size, for now just to see if grasps improve
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
