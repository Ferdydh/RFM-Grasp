import sys
import torch
import trimesh
from pathlib import Path
from typing import Optional, List, Union
import logging

from .data_manager import DataManager, DataSelector
from .acronym import create_gripper_marker

logger = logging.getLogger(__name__)


class GraspVisualizer:
    """Visualizer for grasps and meshes."""

    def __init__(
        self,
        data_root: str,
        selectors: Union[DataSelector, List[DataSelector]],
        cache_dir: Optional[str] = None,
    ):
        """Initialize visualizer with data manager."""
        self.data_manager = DataManager(
            data_root=data_root,
            selectors=selectors,
            cache_dir=cache_dir,
            sdf_size=32,  # SDF size doesn't matter for visualization
        )

    def create_scene(self, mesh_path, num_grasps, gripper_color=[0, 255, 0]):
        """Create a scene with mesh and grasps."""
        logger.info("Starting to create scene...")

        # Load mesh
        logger.info(f"Loading mesh from {mesh_path}")
        mesh = self.data_manager.load_mesh(mesh_path)
        logger.info(f"Loaded mesh with {len(mesh.vertices)} vertices")

        # Get transforms and scale factor
        logger.info("Getting transforms...")
        transforms = self.data_manager.get_transforms(mesh_path)
        logger.info(f"Got {len(transforms)} transforms")
        _, scale_factor, _ = self.data_manager.get_sdf(mesh_path)
        logger.info(f"Got scale factor {scale_factor}")

        # Limit number of grasps if specified
        if num_grasps is not None and num_grasps < len(transforms):
            indices = torch.randperm(len(transforms))[:num_grasps]
            transforms = transforms[indices]
            logger.info(f"Limited to {num_grasps} transforms")

        # Create gripper markers
        logger.info("Creating gripper markers...")
        grasps = []
        for i, transform in enumerate(transforms):
            # Convert to 4x4 transform
            transform_mat = torch.eye(4)
            transform_mat[:3, :3] = transform[:3, :3]
            transform_mat[:3, 3] = transform[:3, 3] * scale_factor

            # Create and transform gripper marker
            grasp_marker = create_gripper_marker(color=gripper_color)
            grasps.append(grasp_marker.apply_transform(transform_mat.numpy()))

            if i % 5 == 0:  # Log every 5th grasp
                logger.info(f"Created {i+1}/{len(transforms)} gripper markers")

        # Scale mesh back to original size
        logger.info("Scaling mesh...")
        vis_mesh = mesh.copy()
        vis_mesh.vertices *= scale_factor

        # Create scene
        logger.info("Creating final scene...")
        scene = trimesh.Scene([vis_mesh] + grasps)
        logger.info("Scene created successfully")
        return scene

    def show_grasps(self, mesh_path=None, num_grasps=20, gripper_color=[0, 255, 0]):
        """Visualize grasps for a mesh."""
        logger.info("Starting show_grasps...")

        # Get mesh path if not specified
        if mesh_path is None:
            mesh_paths = self.data_manager.get_all_mesh_paths()
            if len(mesh_paths) == 0:
                raise ValueError("No meshes found matching the selection criteria!")
            elif len(mesh_paths) > 1:
                raise ValueError(
                    f"Multiple meshes found ({len(mesh_paths)}), please specify which one to visualize!"
                )
            mesh_path = mesh_paths[0]
            logger.info(f"Using mesh path: {mesh_path}")
        elif mesh_path not in self.data_manager.get_all_mesh_paths():
            raise ValueError(f"Mesh {mesh_path} not found in selected data!")

        # Create and show scene
        logger.info("Creating scene...")
        scene = self.create_scene(mesh_path, num_grasps, gripper_color)
        logger.info("Attempting to show scene...")

        # Try alternate viewers if default fails
        try:
            scene.show(viewer="gl")  # Try explicit OpenGL viewer
        except Exception as e:
            logger.error(f"GL viewer failed: {e}")
            try:
                scene.show(viewer="notebook")  # Try notebook viewer
            except Exception as e:
                logger.error(f"Notebook viewer failed: {e}")
                # Fall back to PNG
                logger.info("Falling back to PNG visualization...")
                png = scene.save_image()
                import tempfile
                import os
                import subprocess

                # Save and open with system viewer
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                    f.write(png)
                    temp_path = f.name

                if os.name == "posix":  # macOS or Linux
                    subprocess.run(
                        ["open" if sys.platform == "darwin" else "xdg-open", temp_path]
                    )
                elif os.name == "nt":  # Windows
                    os.startfile(temp_path)

                logger.info(f"Saved visualization to {temp_path}")

    def save_visualization(
        self,
        output_path: Union[str, Path],
        mesh_path: Optional[str] = None,
        num_grasps: Optional[int] = 20,
        gripper_color: List[int] = [0, 255, 0],
        resolution: Optional[tuple] = (1920, 1080),
        camera_params: Optional[dict] = None,
    ) -> None:
        """Save visualization to file.

        Parameters:
            output_path: Path to save the visualization
            mesh_path: Path to specific mesh (optional if only one mesh selected)
            num_grasps: Number of grasps to visualize (None for all)
            gripper_color: RGB color for gripper markers
            resolution: Image resolution as (width, height)
            camera_params: Camera parameters for rendering (optional)
        """
        # Get mesh path if not specified
        if mesh_path is None:
            mesh_paths = self.data_manager.get_all_mesh_paths()
            if len(mesh_paths) == 0:
                raise ValueError("No meshes found matching the selection criteria!")
            elif len(mesh_paths) > 1:
                raise ValueError(
                    f"Multiple meshes found ({len(mesh_paths)}), please specify which one to visualize!"
                )
            mesh_path = mesh_paths[0]
        elif mesh_path not in self.data_manager.get_all_mesh_paths():
            raise ValueError(f"Mesh {mesh_path} not found in selected data!")

        # Create scene
        scene = self.create_scene(mesh_path, num_grasps, gripper_color)

        # Set up camera if provided
        if camera_params is not None:
            scene.camera_transform = camera_params.get(
                "transform", scene.camera_transform
            )
            scene.camera.fov = camera_params.get("fov", scene.camera.fov)

        # Save rendered image
        output_path = Path(output_path)
        try:
            png = scene.save_image(resolution=resolution, visible=True)
            with open(output_path, "wb") as f:
                f.write(png)
        except Exception as e:
            raise RuntimeError(f"Failed to save visualization: {e}")

    def show_all_meshes(
        self,
        num_grasps: Optional[int] = 20,
        gripper_color: List[int] = [0, 255, 0],
    ) -> None:
        """Visualize all selected meshes one by one."""
        mesh_paths = self.data_manager.get_all_mesh_paths()
        if len(mesh_paths) == 0:
            raise ValueError("No meshes found matching the selection criteria!")

        print(
            f"Showing {len(mesh_paths)} meshes. Press any key to advance to next mesh."
        )

        for i, mesh_path in enumerate(mesh_paths, 1):
            print(f"\nShowing mesh {i}/{len(mesh_paths)}: {Path(mesh_path).name}")
            scene = self.create_scene(mesh_path, num_grasps, gripper_color)
            scene.show()
