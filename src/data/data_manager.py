from dataclasses import dataclass
import logging
import os
from pathlib import Path
import pickle

import h5py
import numpy as np
import trimesh

from src.data.util import enforce_trimesh, process_mesh_to_sdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraspCacheEntry:
    """Cache entry for processed grasp data."""

    sdf: np.ndarray
    transforms: np.ndarray  # Scaled transforms
    dataset_mesh_scale: float
    normalization_scale: float
    mesh_path: str


class GraspCache:
    """Cache for processed grasp data."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "grasp_cache.pkl"
        self.cache: dict[str, GraspCacheEntry] = {}

        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Starting with empty cache.")

    def get_or_process(
        self, grasp_filename: str, data_root: str, sdf_size: int
    ) -> GraspCacheEntry:
        """Get cached data or process and cache if not available."""
        if grasp_filename in self.cache:
            logger.info(f"Loading {grasp_filename} from cache")
            return self.cache[grasp_filename]

        logger.info(f"Processing {grasp_filename}")
        grasp_file = os.path.join(data_root, "grasps", grasp_filename)

        with h5py.File(grasp_file, "r") as h5file:
            mesh_fname = h5file["object/file"][()].decode("utf-8")
            dataset_mesh_scale = h5file["object/scale"][()]

            transforms = h5file["grasps"]["transforms"][:]
            grasp_success = h5file["grasps"]["qualities"]["flex"]["object_in_gripper"][
                :
            ]
            transforms = transforms[grasp_success == 1]

        # Load and process mesh
        mesh_path = os.path.join(data_root, mesh_fname)
        mesh = trimesh.load(mesh_path)
        mesh = mesh.apply_scale(dataset_mesh_scale)
        mesh = enforce_trimesh(mesh)

        # Compute SDF and transform grasps
        sdf, normalization_scale, centroid = process_mesh_to_sdf(mesh, sdf_size)

        # Create and cache entry
        entry = GraspCacheEntry(
            sdf=sdf,
            transforms=transforms,
            dataset_mesh_scale=dataset_mesh_scale,
            normalization_scale=normalization_scale,
            mesh_path=mesh_path,
        )
        self.cache[grasp_filename] = entry

        # Save cache
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

        return entry
