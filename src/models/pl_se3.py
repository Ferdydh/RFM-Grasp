from typing import Tuple
import pytorch_lightning as pl
import torch
from torch import Tensor
import trimesh
import wandb
import os


from src.core.config import MLPExperimentConfig
from src.core.visualize import check_collision
from .fm_se3 import FM_SE3
from .wasserstein import wasserstein_distance

# THIS IS IMPORTANT!
os.environ["GEOMSTATS_BACKEND"] = "pytorch"


class FlowMatching(pl.LightningModule):
    def __init__(self, config: MLPExperimentConfig):
        super().__init__()
        self.config = config
        self.se3fm = FM_SE3()

    def compute_loss(
        self,
        so3_inputs,
        r3_inputs,
        prefix: str = "train",
    ) -> Tuple[Tensor, dict[str, Tensor]]:
        loss, loss_dict = self.se3fm.loss(so3_inputs, r3_inputs)
        return loss, {f"{prefix}/{k}": v for k, v in loss_dict.items()}

    def forward(self, so3_input, r3_input, t):
        return self.se3fm.forward(so3_input, r3_input, t)

    def training_step(self, batch, batch_idx):
        print("Training Step")

        (
            so3_input,
            r3_input,
            sdf_input,
            mesh_path,
            dataset_mesh_scale,
            normalization_scale,
        ) = batch

        # self.forward(so3_input, r3_input)

        loss, log_dict = self.compute_loss(so3_input, r3_input, "train")
        if batch_idx % 100 == 0:
            self.log(
                "train/loss",
                loss,
                prog_bar=True,
                batch_size=self.config.data.batch_size,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        print("Validation Step")

        (
            so3_input,
            r3_input,
            sdf_input,
            mesh_path,
            dataset_mesh_scale,
            normalization_scale,
        ) = batch

        # Generate samples using combined SE3FM sampler
        so3_generated, r3_generated = self.se3fm.sample(so3_input, r3_input)

        # Simple L1 loss for validation
        val_loss = torch.mean(torch.abs(r3_generated - r3_input)) + torch.mean(
            torch.abs(so3_generated - so3_input)
        )

        self.log(
            "val/loss", val_loss, prog_bar=True, batch_size=self.config.data.batch_size
        )

        return val_loss

    def configure_optimizers(self):
        # Configure optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.optimizer.lr,
            betas=tuple(self.config.optimizer.betas),
            eps=self.config.optimizer.eps,
            weight_decay=self.config.optimizer.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.scheduler.T_max,
            eta_min=self.config.scheduler.eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.config.checkpoint.monitor,
            },
        }

    def on_train_start(self):
        # Log the original grasp scene
        (
            train_so3_input,
            train_r3_input,
            train_sdf_input,
            train_mesh_path,
            train_dataset_mesh_scale,
            train_normalization_scale,
        ) = self.trainer.train_dataloader.dataset[0]

        (
            val_so3_input,
            val_r3_input,
            val_sdf_input,
            val_mesh_path,
            val_dataset_mesh_scale,
            val_normalization_scale,
        ) = self.trainer.val_dataloaders.dataset[0]

        train_has_collision, train_scene, train_min_distance = check_collision(
            train_so3_input,
            train_r3_input,
            train_mesh_path,
            train_dataset_mesh_scale,
            train_normalization_scale,
        )

        val_has_collision, val_scene, val_min_distance = check_collision(
            val_so3_input,
            val_r3_input,
            val_mesh_path,
            val_dataset_mesh_scale,
            val_normalization_scale,
        )

        self.logger.experiment.log(
            {"train/original_grasp": scene_to_wandb_image(train_scene)}
        )

        self.logger.experiment.log(
            {"val/original_grasp": scene_to_wandb_image(val_scene)}
        )


def scene_to_wandb_image(scene: trimesh.Scene) -> wandb.Image:
    """
    Log a colored front view of a trimesh scene using PyVista's off-screen rendering.
    Returns a low-res wandb.Image for basic visualization.
    """
    import pyvista as pv
    import numpy as np

    # Convert trimesh scene to PyVista
    plotter = pv.Plotter(off_screen=True)

    # Add each mesh from the scene
    for geometry in scene.geometry.values():
        if hasattr(geometry, "vertices") and hasattr(geometry, "faces"):
            # Convert trimesh to PyVista
            mesh = pv.PolyData(
                geometry.vertices,
                np.hstack([[3] + face.tolist() for face in geometry.faces]),
            )

            # Handle color
            if hasattr(geometry, "visual") and hasattr(geometry.visual, "face_colors"):
                face_colors = geometry.visual.face_colors
                if face_colors is not None:
                    # Convert RGBA to RGB if needed
                    if face_colors.shape[1] == 4:
                        face_colors = face_colors[:, :3]
                    mesh.cell_data["colors"] = face_colors
                    plotter.add_mesh(mesh, scalars="colors", rgb=True)
            else:
                # Default color if no colors specified
                plotter.add_mesh(mesh, color="lightgray")

    # Set a very low resolution
    plotter.window_size = [1024, 1024]

    # Get the image array
    img_array = plotter.screenshot(return_img=True)

    # Close the plotter
    plotter.close()

    return wandb.Image(img_array)
