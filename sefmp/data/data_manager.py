import os
import glob
import h5py
import torch
import trimesh
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Set
from collections import defaultdict
import time

from .data_structures import DataSelector, SDFCache, compute_mesh_hash

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """Manager for mesh, SDF, and grasp data with file-based SDF caching."""

    def __init__(
        self,
        data_root: str,
        selectors: Union[DataSelector, List[DataSelector]],
        cache_dir: Optional[str] = None,
        sdf_size: int = 32,
        compute_sdf: bool = False,
    ):
        start_time = time.time()
        logger.info(f"Initializing DataManager with root: {data_root}")

        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_root / "sdf_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sdf_size = sdf_size
        self.compute_sdf = compute_sdf

        # Convert single selector to list
        self.selectors = (
            [selectors] if isinstance(selectors, DataSelector) else selectors
        )

        # Find all available mesh and grasp files
        self.mesh_pattern = str(self.data_root / "meshes" / "**" / "*.obj")
        self.grasp_pattern = str(self.data_root / "grasps" / "*.h5")

        all_meshes = glob.glob(self.mesh_pattern, recursive=True)
        all_grasps = glob.glob(self.grasp_pattern)

        logger.info(f"Found {len(all_meshes)} meshes and {len(all_grasps)} grasp files")

        # Filter data and create mappings
        self.mesh_to_grasps: Dict[str, List[str]] = {}
        self.mesh_to_transforms: Dict[str, torch.Tensor] = {}

        # Build lookup tables and filter data
        self._filter_data_optimized(all_meshes, all_grasps)
        self._load_all_transforms()

        logger.info(
            f"DataManager initialization took {time.time() - start_time:.2f} seconds"
        )

    def _filter_data_optimized(self, meshes: List[str], grasps: List[str]) -> None:
        """Filter meshes and grasps based on selectors with optimized matching."""
        start_time = time.time()
        logger.info("Starting optimized data filtering...")

        # Pre-process grasps to avoid repeated filename parsing
        processed_grasps: Dict[str, List[str]] = defaultdict(list)

        for grasp in grasps:
            grasp_filename = os.path.basename(grasp)
            if "_" not in grasp_filename:
                continue

            try:
                item_name, obj_id, file_id = grasp_filename.split("_")
                processed_grasps[obj_id].append(grasp)
            except ValueError:
                continue

        # Process meshes and find matches
        for mesh in meshes:
            matching_grasps = []

            # Check each grasp key against the mesh path
            for key, grasp_list in processed_grasps.items():
                if key not in mesh:
                    continue

                # Only check selectors if the basic path match succeeds
                for grasp in grasp_list:
                    if any(
                        selector.matches(mesh, grasp) for selector in self.selectors
                    ):
                        matching_grasps.append(grasp)

            if matching_grasps:
                self.mesh_to_grasps[mesh] = matching_grasps

        logger.info(f"Selected {len(self.mesh_to_grasps)} meshes based on criteria")
        logger.info(f"Data filtering took {time.time() - start_time:.2f} seconds")

    def _load_all_transforms(self) -> None:
        """Load and scale transforms for all selected meshes."""
        start_time = time.time()
        logger.info("Loading transforms for all selected meshes...")

        for mesh_path, grasp_files in self.mesh_to_grasps.items():
            try:
                # Get scale factor from mesh
                mesh = self.load_mesh(mesh_path)
                scale_factor = np.max(np.abs(mesh.vertices - mesh.centroid))

                transforms_list = []
                for grasp_file in grasp_files:
                    with h5py.File(grasp_file, "r") as h5file:
                        transforms = h5file["grasps"]["transforms"][:]
                        success = h5file["grasps"]["qualities"]["flex"][
                            "object_in_gripper"
                        ][:]
                        transforms = transforms[success == 1]

                        # Scale translations in one operation
                        transforms[:, :3, 3] /= scale_factor
                        transforms_list.append(transforms)

                if transforms_list:
                    self.mesh_to_transforms[mesh_path] = torch.tensor(
                        np.concatenate(transforms_list), dtype=torch.float32
                    )

            except Exception as e:
                logger.error(f"Error loading transforms for {mesh_path}: {e}")
                raise

        logger.info(f"Transform loading took {time.time() - start_time:.2f} seconds")

    def load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        """Load a mesh from file."""
        start_time = time.time()
        logger.info(f"Loading mesh from: {mesh_path}")

        try:
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(
                    [
                        trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                        for m in mesh.geometry.values()
                    ]
                )

            logger.info(f"Mesh loading took {time.time() - start_time:.2f} seconds")
            return mesh

        except Exception as e:
            logger.error(f"Error loading mesh: {e}")
            raise

    def get_sdf(self, mesh_path: str) -> tuple[np.ndarray, float, np.ndarray]:
        """Get SDF data for a mesh, computing and caching if necessary."""
        if not self.compute_sdf:
            return (
                np.zeros((self.sdf_size, self.sdf_size, self.sdf_size)),
                1.0,
                np.array([0.0, 0.0, 0.0]),
            )

        start_time = time.time()
        logger.info(f"Getting SDF for mesh: {mesh_path}")

        cache_path = self.cache_dir / f"{compute_mesh_hash(mesh_path)}.pkl"

        # Try to load from cache
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.mesh_hash == compute_mesh_hash(mesh_path):
                    logger.info(
                        f"Cache hit - loaded in {time.time() - start_time:.2f}s"
                    )
                    return cached.sdf, cached.scale_factor, cached.centroid
            except Exception as e:
                logger.warning(f"Cache load failed for {mesh_path}: {e}")

        # Compute if needed
        mesh = self.load_mesh(mesh_path)
        cache_data = self._compute_sdf(mesh, mesh_path)

        # Save to cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.error(f"Cache save failed for {mesh_path}: {e}")

        return cache_data.sdf, cache_data.scale_factor, cache_data.centroid

    def _compute_sdf(self, mesh: trimesh.Trimesh, mesh_path: str) -> SDFCache:
        """Compute SDF for a mesh."""
        if not self.compute_sdf:
            return SDFCache(
                sdf=np.zeros((self.sdf_size, self.sdf_size, self.sdf_size)),
                scale_factor=1.0,
                centroid=np.array([0.0, 0.0, 0.0]),
                mesh_hash=compute_mesh_hash(mesh_path),
            )

        start_time = time.time()
        logger.info(f"Computing SDF for mesh: {mesh_path}")

        try:
            import mesh2sdf

            # Process mesh
            centroid = mesh.centroid
            vertices = (mesh.vertices - centroid) / np.max(
                np.abs(mesh.vertices - centroid)
            )

            # Compute SDF efficiently
            faces = mesh.faces.astype(np.uint32)
            sdf = mesh2sdf.compute(vertices, faces, self.sdf_size, fix=True)

            logger.info(f"SDF computation took {time.time() - start_time:.2f} seconds")

            return SDFCache(
                sdf=sdf,
                scale_factor=np.max(np.abs(mesh.vertices - centroid)),
                centroid=centroid,
                mesh_hash=compute_mesh_hash(mesh_path),
            )
        except Exception as e:
            logger.error(f"Error computing SDF: {e}")
            raise

    def get_transforms(self, mesh_path: str) -> torch.Tensor:
        """Get transforms for a mesh."""
        return self.mesh_to_transforms[mesh_path]

    def get_all_mesh_paths(self) -> List[str]:
        """Get list of all selected mesh paths."""
        return list(self.mesh_to_grasps.keys())

    def get_grasp_files(self, mesh_path: str) -> List[str]:
        """Get grasp files associated with a mesh."""
        return self.mesh_to_grasps.get(mesh_path, [])
