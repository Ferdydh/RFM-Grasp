import os
import glob
import h5py
import torch
import trimesh
import mesh2sdf
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from collections import defaultdict


@dataclass
class DataSelector:
    """
    Flexible data selection criteria for grasps dataset.

    Examples:
        # Select by specific grasp ID
        DataSelector(grasp_id="0.017127096456853612")

        # Select by object ID
        DataSelector(object_id="abc86fa217ffeeeab3c769e22341ffb4")

        # Select by item name
        DataSelector(item_name="CerealBox")
    """

    grasp_id: Optional[str] = None
    object_id: Optional[str] = None
    item_name: Optional[str] = None

    def matches(self, mesh_path: str, grasp_path: str) -> bool:
        """Check if paths match the selection criteria."""
        # Extract information from grasp path
        grasp_filename = os.path.basename(grasp_path)
        if "_" in grasp_filename:
            item_name, obj_id, grasp_value = grasp_filename.split("_")
            # Remove .h5 extension and get just the grasp value
            grasp_value = grasp_value.replace(".h5", "")
        else:
            return False

        # Check grasp ID
        if self.grasp_id and self.grasp_id != grasp_value:
            return False

        # Check object ID
        if self.object_id and self.object_id != obj_id:
            return False

        # Check item name
        if self.item_name and self.item_name != item_name:
            return False

        return True


def process_mesh_to_sdf(mesh_path: str, size: int = 32) -> Tuple[np.ndarray, float]:
    """Process a mesh to SDF with consistent scaling and centering."""
    # Load mesh
    mesh = trimesh.load(mesh_path)

    # Convert scene to mesh if necessary
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in mesh.geometry.values()
            )
        )

    # Center the vertices
    centroid = mesh.centroid
    vertices = mesh.vertices - centroid

    # Compute scale factor based on max absolute coordinate
    scale_factor = np.max(np.abs(vertices))

    # Scale vertices to [-1, 1]
    vertices = vertices / scale_factor
    faces = mesh.faces

    # Ensure correct dtype for mesh2sdf
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.uint32)

    # First compute raw SDF to see its range
    raw_sdf = mesh2sdf.core.compute(vertices, faces, size)
    abs_sdf = np.abs(raw_sdf)
    print(f"SDF range before abs: {raw_sdf.min():.6f} to {raw_sdf.max():.6f}")
    print(f"SDF range after abs: {abs_sdf.min():.6f} to {abs_sdf.max():.6f}")

    # Use a level value within the range of the absolute SDF
    level = (abs_sdf.min() + abs_sdf.max()) / 2

    # Compute SDF with our chosen level
    sdf = mesh2sdf.compute(vertices, faces, size, fix=True, level=level)

    return sdf, scale_factor


def compute_scale_factor(mesh: Union[trimesh.Trimesh, trimesh.Scene]) -> float:
    """Compute scale factor to normalize mesh to [-1, 1]."""
    if isinstance(mesh, trimesh.Scene):
        extents = np.array([m.extents for m in mesh.geometry.values()]).max(axis=0)
        return np.max(extents)
    else:
        return np.max(mesh.extents)


def _load_mesh(mesh_path: str) -> Tuple[trimesh.Trimesh, float]:
    """Load mesh and compute scale factor."""
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in mesh.geometry.values()
            )
        )
    scale_factor = compute_scale_factor(mesh)
    return mesh, scale_factor


class GraspDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        selectors: Union[DataSelector, List[DataSelector]],
        split: str = "train",
        num_samples: Optional[int] = None,
        sdf_size: int = 32,
    ):
        """
        Args:
            data_root: Root directory containing meshes and grasps folders
            selectors: Single DataSelector or list of DataSelectors
            split: One of 'train', 'val', 'test'
            num_samples: Optional limit on total number of samples
            sdf_size: Size of the SDF grid (default: 32)
        """
        self.data_root = data_root
        self.selectors = (
            [selectors] if isinstance(selectors, DataSelector) else selectors
        )
        self.split = split
        self.num_samples = num_samples
        self.sdf_size = sdf_size

        # Get all meshes and grasps
        mesh_pattern = os.path.join(data_root, "meshes/**/*.obj")
        grasp_pattern = os.path.join(data_root, "grasps/*.h5")

        self.meshes = glob.glob(mesh_pattern, recursive=True)
        self.grasps = glob.glob(grasp_pattern)

        # Group grasps by mesh (object_id)
        self.mesh_to_grasps = self._create_mesh_grasp_mapping()

        # Process all unique meshes to SDFs and get their scales
        self.mesh_to_sdf = {}
        self._compute_all_sdfs()

        # Load all transforms and scale translations
        self.mesh_to_transforms = {}
        self._load_all_transforms()

        # Create flat index for accessing data
        self._create_index()

    def _compute_all_sdfs(self):
        """Compute SDFs for all unique meshes."""
        for mesh_path in self.mesh_to_grasps.keys():
            sdf, _ = process_mesh_to_sdf(mesh_path, self.sdf_size)
            self.mesh_to_sdf[mesh_path] = sdf

    def _load_all_transforms(self):
        """Load transforms and scale translations according to object scale."""
        for mesh_path, grasp_files in self.mesh_to_grasps.items():
            # Load mesh and get scale factor
            mesh, scale_factor = _load_mesh(mesh_path)

            transforms_list = []
            for grasp_file in grasp_files:
                with h5py.File(grasp_file, "r") as h5file:
                    transforms = h5file["grasps"]["transforms"][:]
                    # Scale translations to match normalized space
                    for transform in transforms:
                        transform[:3, 3] = transform[:3, 3] / scale_factor * 2.0
                    transforms_list.extend(transforms)

            self.mesh_to_transforms[mesh_path] = torch.tensor(
                transforms_list, dtype=torch.float32
            )

    def _create_mesh_grasp_mapping(self) -> Dict[str, List[str]]:
        """Create mapping from mesh paths to their corresponding grasp files."""
        mesh_to_grasps = defaultdict(list)

        for grasp in self.grasps:
            grasp_filename = os.path.basename(grasp)
            if "_" not in grasp_filename:
                continue

            item_name, obj_id, _ = grasp_filename.split("_", 2)

            # Find corresponding mesh
            matching_meshes = [m for m in self.meshes if obj_id in m and item_name in m]

            if not matching_meshes:
                continue

            mesh = matching_meshes[0]

            # Check if any selector matches
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
        """
        Returns:
            Tuple of (rotation_matrix, translation_vector, sdf)
            rotation_matrix: shape (3, 3)
            translation_vector: shape (3,) - scaled to match normalized object space
            sdf: shape (size, size, size)
        """
        mesh_path, transform_idx = self.index[idx]

        transform = self.mesh_to_transforms[mesh_path][transform_idx]
        rotation = transform[:3, :3]
        translation = transform[:3, 3]  # Already scaled in _load_all_transforms

        sdf = torch.tensor(self.mesh_to_sdf[mesh_path], dtype=torch.float32)

        return rotation, translation, sdf


class GraspDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        selectors: Union[DataSelector, List[DataSelector]],
        batch_size: int = 32,
        num_workers: int = 4,
        num_samples: Optional[int] = None,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        super().__init__()
        self.data_root = data_root
        self.selectors = selectors
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.train_val_test_split = train_val_test_split

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = GraspDataset(
                self.data_root,
                self.selectors,
                split="train",
                num_samples=self.num_samples,
            )

            self.val_dataset = GraspDataset(
                self.data_root,
                self.selectors,
                split="val",
                num_samples=self.num_samples,
            )

        if stage == "test" or stage is None:
            self.test_dataset = GraspDataset(
                self.data_root,
                self.selectors,
                split="test",
                num_samples=self.num_samples,
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


# # Example usage:
# if __name__ == "__main__":
#     # Example 1: Select specific grasp
#     selector1 = DataSelector(grasp_id="0.017127096456853612")

#     # Example 2: Select multiple items
#     selectors = [
#         DataSelector(item_name="CerealBox"),
#         DataSelector(item_name="Mug")
#     ]

#     # Example 3: Select by object ID
#     selector3 = DataSelector(object_id="abc86fa217ffeeeab3c769e22341ffb4")

#     # Initialize DataModule with multiple selectors
#     grasp_data = GraspDataModule(
#         data_root="data",
#         selectors=selectors,  # Using list of selectors
#         batch_size=32,
#         num_samples=1000  # Optional: limit total samples
#     )

#     # Set up the data module
#     grasp_data.setup()

#     # Get data loaders
#     train_loader = grasp_data.train_dataloader()
#     val_loader = grasp_data.val_dataloader()
#     test_loader = grasp_data.test_dataloader()

#     # Example iteration
#     for rotation, translation, sdf in train_loader:
#         print(f"Rotation shape: {rotation.shape}")      # (batch_size, 3, 3)
#         print(f"Translation shape: {translation.shape}") # (batch_size, 3)
#         print(f"SDF shape: {sdf.shape}")               # (batch_size, size, size, size)
#         break
