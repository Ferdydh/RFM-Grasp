from dataclasses import dataclass
import logging
import os
from pathlib import Path
import pickle
import sys
import h5py
import numpy as np
import trimesh
from typing import Optional,Tuple
from src.data.util import enforce_trimesh, process_mesh_to_sdf
import concurrent.futures
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
    centroid: np.ndarray 


class GraspCache:
    """Cache for processed grasp data."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "grasp_cache.pkl"
        self.cache: dict[str, GraspCacheEntry] = {}
        self._load()
        
        # if self.cache_file.exists():
        #     try:
        #         with open(self.cache_file, "rb") as f:
        #             self.cache = pickle.load(f)
        #     except Exception as e:
        #         logger.warning(f"Failed to load cache: {e}. Starting with empty cache.")

    def worker_process_one_file(self,arg_tuple):
        filename, data_root, sdf_size = arg_tuple
        
        # Instead of `cache.get_or_process`, directly open the file, do the SDF, etc.
        # Return the processed data in memory (entry-like).
        
        full_path = os.path.join(data_root, "grasps", filename)
        with h5py.File(full_path, "r") as h5file:
            transforms = h5file["grasps"]["transforms"][:]
            grasp_success = h5file["grasps"]["qualities"]["flex"]["object_in_gripper"][:]
            transforms = transforms[grasp_success == 1]
            if transforms.size == 0:
                return filename, None
            mesh_fname = h5file["object/file"][()].decode("utf-8")
            dataset_mesh_scale = h5file["object/scale"][()]

        mesh_path = os.path.join(data_root, mesh_fname)
        # Do heavy-lifting here: load mesh, compute SDF, etc.
        mesh = trimesh.load(mesh_path)
        mesh.apply_scale(dataset_mesh_scale)
        mesh = enforce_trimesh(mesh)
        sdf, normalization_scale, centroid = process_mesh_to_sdf(mesh, sdf_size)

        # Adjust transforms
        transforms[:, :3, 3] -= centroid

        # Return enough info for the main process to build a GraspCacheEntry
        return filename, {
            "sdf": sdf,
            "transforms": transforms,
            "dataset_mesh_scale": dataset_mesh_scale,
            "normalization_scale": normalization_scale,
            "mesh_path": mesh_path,
            "centroid": centroid,
        }
    
    def build_cache_in_main(self, grasp_files, data_root, sdf_size, max_workers=4):
        """
        1. Load existing cache from disk (if exists).
        2. Parallel process each file if needed.
        3. Merge results into in-memory cache dict.
        4. Write cache to disk once at the end.
        """
        # Step 1: Load existing cache
        self._load()

        # Filter out files that are already in the cache
        results = [None] * len(grasp_files)
        
        files_to_process = [f for f in grasp_files if f not in self.cache]

        # Step 2: Use concurrency for heavy-lifting

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map function to each file
            futures = {
                executor.submit(self.worker_process_one_file, (filename, data_root, sdf_size)): filename
                for filename in files_to_process
            }

            # Step 3: Merge results back into the cache in the MAIN PROCESS ONLY
            for future in concurrent.futures.as_completed(futures):
                filename = futures[future]
                try:
                    filename, result_dict = future.result()
                    # If result_dict is None, skip
                    if result_dict is not None:
                        # create the entry
                        entry = GraspCacheEntry(**result_dict)
                        self.cache[filename] = entry
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
        
        # Step 4: Write once
        self._save()

    def _load(self):
        """Load entire cache from pickle once. (Main process only)"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Starting empty cache.")

    def _save(self):
        """Write entire cache dict to pickle once. (Main process only)"""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")



    def get_or_process(
        self, grasp_filename: str, data_root: str, sdf_size: int
    ) -> Optional[GraspCacheEntry]:
        """Get cached data or process and cache if not available."""
        if grasp_filename in self.cache:
            logger.info(f"Loading {grasp_filename} from cache")
            return self.cache[grasp_filename]

        logger.info(f"Processing {grasp_filename}")
        
        grasp_file = os.path.join(data_root, "grasps", grasp_filename)

        with h5py.File(grasp_file, "r") as h5file:
            transforms = h5file["grasps"]["transforms"][:]
            grasp_success = h5file["grasps"]["qualities"]["flex"]["object_in_gripper"][
                :
            ]
            transforms = transforms[grasp_success == 1]
            if transforms.size == 0:
                return None
            mesh_fname = h5file["object/file"][()].decode("utf-8")
            dataset_mesh_scale = h5file["object/scale"][()]

            
            
        # Load and process mesh
        mesh_path = os.path.join(data_root, mesh_fname)
        mesh = trimesh.load(mesh_path)
        mesh = mesh.apply_scale(dataset_mesh_scale)
        mesh = enforce_trimesh(mesh)
        # Compute SDF and transform grasps
        sdf, normalization_scale, centroid = process_mesh_to_sdf(mesh, sdf_size)
        print(transforms.shape,centroid.shape)
        transforms[:, :3, 3] -= centroid
        # Create and cache entry
        entry = GraspCacheEntry(
            sdf=sdf,
            transforms=transforms,
            dataset_mesh_scale=dataset_mesh_scale,
            normalization_scale=normalization_scale,
            mesh_path=mesh_path,
            centroid=centroid,
        )
        self.cache[grasp_filename] = entry

        # Save cache
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

        return entry

    @staticmethod
    def process_one_file(args: Tuple[str, str, int]) -> Optional[Tuple[str, Optional[GraspCacheEntry], np.ndarray, np.ndarray, int]]:
        """
        Worker function to process a single .h5 file.
        Returns (filename, entry, local_min, local_max, num_grasps)
        or None if something failed badly.

        - entry can be None if the file has 0 successful transforms.
        - local_min, local_max are 3D vectors from the transforms.
        - num_grasps = number of transforms.
        """
        filename, data_root, sdf_size = args

        try:
            grasp_file = os.path.join(data_root, "grasps", filename)
            with h5py.File(grasp_file, "r") as h5file:
                transforms = h5file["grasps"]["transforms"][:]
                success = h5file["grasps"]["qualities"]["flex"]["object_in_gripper"][:]
                transforms = transforms[success == 1]
                if len(transforms) == 0:
                    # We'll return an entry=None to indicate no valid grasps
                    return (filename, None, None, None, 0)

                mesh_fname = h5file["object/file"][()].decode("utf-8")
                dataset_mesh_scale = h5file["object/scale"][()]
            
            mesh_path = os.path.join(data_root, mesh_fname)
            mesh = trimesh.load(mesh_path)
            mesh.apply_scale(dataset_mesh_scale)
            mesh = enforce_trimesh(mesh)

            # Compute SDF
            sdf, normalization_scale, centroid = process_mesh_to_sdf(mesh, sdf_size)

            # Adjust transforms by centroid
            transforms[:, :3, 3] -= centroid

            # local min/max from these transforms
            local_min = np.min(transforms[:, :3, 3], axis=0)
            local_max = np.max(transforms[:, :3, 3], axis=0)
            num_grasps = len(transforms)

            entry = GraspCacheEntry(
                sdf=sdf,
                transforms=transforms,
                dataset_mesh_scale=dataset_mesh_scale,
                normalization_scale=normalization_scale,
                mesh_path=mesh_path,
                centroid=centroid,
            )
            return (filename, entry, local_min, local_max, num_grasps)

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            return None




