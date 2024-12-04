from dataclasses import dataclass
import numpy as np
from typing import Optional
import hashlib
import os


@dataclass
class DataSelector:
    """Selection criteria for filtering grasp dataset."""

    grasp_id: Optional[str] = None
    object_id: Optional[str] = None
    item_name: Optional[str] = None

    def matches(self, mesh_path: str, grasp_path: str) -> bool:
        """Check if paths match the selection criteria."""
        grasp_filename = os.path.basename(grasp_path)
        if "_" not in grasp_filename:
            return False

        item_name, obj_id, grasp_value = grasp_filename.split("_")
        grasp_value = grasp_value.replace(".h5", "")

        # Short-circuit if any criteria don't match
        if self.grasp_id and self.grasp_id != grasp_value:
            return False
        if self.object_id and self.object_id != obj_id:
            return False
        if self.item_name and self.item_name != item_name:
            return False

        # Verify mesh path contains matching identifiers
        if obj_id not in mesh_path or item_name not in mesh_path:
            return False

        return True


@dataclass
class SDFCache:
    """Data structure for cached SDF and associated metadata."""

    sdf: np.ndarray
    scale_factor: float
    centroid: np.ndarray
    mesh_hash: str


def compute_mesh_hash(mesh_path: str) -> str:
    """Compute a hash of the mesh file contents using SHA-256."""
    sha256_hash = hashlib.sha256()
    with open(mesh_path, "rb") as f:
        # Read the file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
