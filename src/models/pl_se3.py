import torch.nn.functional as F
from einops import rearrange
from scipy.spatial.transform import Rotation
from typing import Tuple, Dict, Any
import pytorch_lightning as pl
import torch
from torch import Tensor
import wandb

from src.core.config import BaseExperimentConfig
from src.core.visualize import check_collision
from src.models.util import sample_location_and_conditional_flow, scene_to_wandb_image
from src.models.fm_se3 import FM_SE3


class FlowMatching(pl.LightningModule):
    """Flow Matching model combining SO3 and R3 manifold learning with synchronized time sampling."""

    def __init__(self, config: BaseExperimentConfig):
        super().__init__()
        self.config = config
        self.se3fm = FM_SE3(config)

        # TODO use config
        self.sigma_min: float = 1e-4
        self.save_hyperparameters()

    def compute_loss(
        self,
        so3_inputs: Tensor,
        r3_inputs: Tensor,
        prefix: str = "train",
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute combined loss for both manifolds with synchronized time sampling.

        Args:
            so3_inputs: Target SO3 matrices [batch, 3, 3]
            r3_inputs: Target R3 points [batch, 3]
            prefix: Prefix for logging metrics

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Sample synchronized time points for both manifolds
        t = torch.rand(so3_inputs.shape[0], device=so3_inputs.device)

        # SO3 computation
        x0_so3 = torch.tensor(
            Rotation.random(so3_inputs.size(0)).as_matrix(), device=so3_inputs.device
        )

        # Sample location and flow for SO3
        xt_so3, ut_so3 = sample_location_and_conditional_flow(x0_so3, so3_inputs, t)

        # Get velocity prediction for SO3
        xt_flat = rearrange(xt_so3, "b c d -> b (c d)", c=3, d=3)
        vt_so3 = self.se3fm.so3fm.forward(xt_flat, t[:, None])
        vt_so3 = rearrange(vt_so3, "b (c d) -> b c d", c=3, d=3)

        # Compute SO3 loss using Riemannian metric
        r = torch.transpose(xt_so3, dim0=-2, dim1=-1) @ (vt_so3 - ut_so3)
        norm = -torch.diagonal(r @ r, dim1=-2, dim2=-1).sum(dim=-1) / 2
        so3_loss = torch.mean(norm, dim=-1)

        # R3 computation with same time points
        t_expanded = t.unsqueeze(-1)  # [batch, 1]
        noise = torch.randn_like(r3_inputs)

        # Compute noisy sample and optimal flow for R3
        x_t_r3 = (
            1 - (1 - self.sigma_min) * t_expanded
        ) * noise + t_expanded * r3_inputs
        optimal_flow = r3_inputs - (1 - self.sigma_min) * noise

        # Get predicted flow for R3
        predicted_flow = self.se3fm.r3fm.forward(x_t_r3, t_expanded)

        # Compute R3 MSE loss
        r3_loss = F.mse_loss(predicted_flow, optimal_flow)

        total_loss = so3_loss + r3_loss

        loss_dict = {
            f"{prefix}/so3_loss": so3_loss,
            f"{prefix}/r3_loss": r3_loss,
            f"{prefix}/loss": total_loss,
        }

        return total_loss, loss_dict

    def forward(
        self, so3_input: Tensor, r3_input: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through the model."""
        return self.se3fm.forward(so3_input, r3_input, t)

    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Training step implementation."""
        so3_input, r3_input, *_ = batch

        # Repeat the batch if its size is smaller than the configured batch size
        if so3_input.shape[0] < self.config.data.batch_size:
            repeat_factor = (self.config.data.batch_size // so3_input.shape[0]) + 1
            so3_input = so3_input.repeat(repeat_factor, 1, 1)[:self.config.data.batch_size]
            r3_input = r3_input.repeat(repeat_factor, 1)[:self.config.data.batch_size]

        loss, log_dict = self.compute_loss(so3_input, r3_input, "train")

        self.log_dict(
            log_dict,
            prog_bar=True,
            batch_size=self.config.data.batch_size,
        )

        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, Tensor]:
        """Validation step implementation."""
        so3_input, r3_input, *_ = batch

        with torch.enable_grad():
            loss, log_dict = self.compute_loss(so3_input, r3_input, "val")

        # Log validation metrics
        self.log_dict(
            log_dict,
            prog_bar=True,
            batch_size=self.config.data.batch_size,
        )

        return log_dict

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
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

    def on_train_start(self) -> None:
        """Setup logging of initial grasp scenes on training start."""
        for prefix, dataset in [
            ("train", self.trainer.train_dataloader.dataset),
            ("val", self.trainer.val_dataloaders.dataset),
        ]:
            (
                so3_input,
                r3_input,
                _,  # sdf_input
                mesh_path,
                dataset_mesh_scale,
                normalization_scale,
            ) = dataset[0]

            has_collision, scene, min_distance = check_collision(
                so3_input,
                r3_input,
                mesh_path,
                dataset_mesh_scale,
            )

            gripper_transform = torch.eye(4)
            gripper_transform[:3, :3] = so3_input[:3, :3]
            gripper_transform[:3, 3] = r3_input.squeeze()

            gripper_transform = wandb.Table(
                data=gripper_transform.cpu().numpy().tolist(),
                columns=["rot1", "rot2", "rot3", "tr"],
            )

            self.logger.experiment.log(
                {
                    f"{prefix}/original_grasp": scene_to_wandb_image(scene),
                    f"{prefix}/original_grasp_transform": gripper_transform,
                }
            )

    def on_train_end(self):
        (
            so3_input,
            r3_input,
            _,  # sdf_input
            mesh_path,
            dataset_mesh_scale,
            normalization_scale,
        ) = self.trainer.train_dataloader.dataset[0]

        print("Training end")
        # print("so3_input", so3_input)
        # print("r3_input", r3_input)
        # print("mesh_path", mesh_path)
        # print("dataset_mesh_scale", dataset_mesh_scale)

        so3_output, r3_output = self.se3fm.sample(
            so3_input, r3_input, self.config.logging.num_samples_to_visualize
        )

        # print("so3_output", so3_output)
        # print("r3_output", r3_output)

        batch_size = so3_output.shape[0]  # Assuming first dimension is batch size

        for batch_idx in range(batch_size):
            # Extract single sample from batch
            so3_sample = so3_output[batch_idx]  # Shape: (dim,)
            r3_sample = r3_output[batch_idx]  # Shape: (dim,)

            # Check collision for this sample
            has_collision, scene, min_distance = check_collision(
                so3_sample,
                r3_sample,
                mesh_path,
                dataset_mesh_scale,
            )

            gripper_transform = torch.eye(4)
            gripper_transform[:3, :3] = so3_sample[:3, :3]
            gripper_transform[:3, 3] = r3_sample.squeeze()

            gripper_transform = wandb.Table(
                data=gripper_transform.cpu().numpy().tolist(),
                columns=["rot1", "rot2", "rot3", "tr"],
            )

            # Log each sample's visualization
            self.logger.experiment.log(
                {
                    f"val/generated_grasp_{batch_idx}": scene_to_wandb_image(scene),
                    f"val/generated_grasp_{batch_idx}_transform": gripper_transform,
                }
            )
