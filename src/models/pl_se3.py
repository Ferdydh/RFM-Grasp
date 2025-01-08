import torch.nn.functional as F
from einops import rearrange
from scipy.spatial.transform import Rotation
from typing import Tuple, Dict, Any
import pytorch_lightning as pl
import torch
from torch import Tensor
import wandb
from src.data.util import denormalize_translation

from src.core.config import BaseExperimentConfig
from src.core.visualize import (
    check_collision,
    check_collision_multiple_grasps,
    scene_to_wandb_image,
)
from src.models.flow import sample, sample_location_and_conditional_flow

from src.models.velocity_mlp import VelocityNetwork
from src.models.wasserstein import wasserstein_distance


class FlowMatching(pl.LightningModule):
    """Flow Matching model combining SO3 and R3 manifold learning with synchronized time sampling."""

    def __init__(self, config: BaseExperimentConfig):
        super().__init__()
        self.config = config
        self.model = VelocityNetwork()

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

        # R3 computation with same time points
        t_expanded = t.unsqueeze(-1)  # [batch, 1]
        noise = torch.randn_like(r3_inputs)

        # Get predicted flow for R3
        x_t_r3 = (
            1 - (1 - self.sigma_min) * t_expanded
        ) * noise + t_expanded * r3_inputs

        vt_so3, predicted_flow = self.forward(xt_flat, x_t_r3, t_expanded)

        vt_so3 = rearrange(vt_so3, "b (c d) -> b c d", c=3, d=3)

        # Compute SO3 loss using Riemannian metric
        r = torch.transpose(xt_so3, dim0=-2, dim1=-1) @ (vt_so3 - ut_so3)
        norm = -torch.diagonal(r @ r, dim1=-2, dim2=-1).sum(dim=-1) / 2
        so3_loss = torch.mean(norm, dim=-1)

        # Compute noisy sample and optimal flow for R3
        optimal_flow = r3_inputs - (1 - self.sigma_min) * noise

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
        return self.model.forward(so3_input, r3_input, t)

    def _adjust_batch_size(
        self, so3_input: Tensor, r3_input: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Adjust batch size by repeating if necessary."""
        if so3_input.shape[0] < self.config.data.batch_size:
            repeat_factor = (self.config.data.batch_size // so3_input.shape[0]) + 1
            so3_input = so3_input.repeat(repeat_factor, 1, 1)[
                : self.config.data.batch_size
            ]
            r3_input = r3_input.repeat(repeat_factor, 1)[: self.config.data.batch_size]
        return so3_input, r3_input

    def _calculate_wasserstein_metrics(
        self, so3_input: Tensor, r3_input: Tensor
    ) -> Dict[str, float]:
        """Calculate Wasserstein distances between input and generated samples."""
        # Generate samples
        so3_output, r3_output = sample(
            self.model, r3_input.device, num_samples=self.config.data.batch_size
        )

        # Compute Wasserstein distances
        w_dist_so3 = wasserstein_distance(
            so3_input, so3_output, space="so3", method="exact", power=2
        )
        w_dist_r3 = wasserstein_distance(
            r3_input, r3_output, space="r3", method="exact", power=2
        )

        return {"wasserstein_so3": w_dist_so3, "wasserstein_r3": w_dist_r3}

    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Training step implementation."""
        so3_input, r3_input, *_ = batch
        so3_input_expanded, r3_input_expanded = self._adjust_batch_size(
            so3_input, r3_input
        )

        loss, log_dict = self.compute_loss(
            so3_input_expanded, r3_input_expanded, "train"
        )

        # Calculate Wasserstein distance every 100 epochs
        current_epoch = self.current_epoch
        # TODO: make once each n epochs a parameter
        # TODO: instead of using just a batch
        if current_epoch > 0 and current_epoch % 100 == 0:
            # creates config batch size amount of noise and compare the resulting points with them
            wasserstein_metrics = self._calculate_wasserstein_metrics(
                so3_input, r3_input
            )
            log_dict.update(
                {
                    f"train/wasserstein_so3": wasserstein_metrics["wasserstein_so3"],
                    f"train/wasserstein_r3": wasserstein_metrics["wasserstein_r3"],
                }
            )

        self.log_dict(
            log_dict,
            prog_bar=True,
            batch_size=self.config.data.batch_size,
        )

        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, Tensor]:
        """Validation step implementation."""
        so3_input, r3_input, *_ = batch

        # Repeat dataset until target batch size is obtained
        so3_input, r3_input = self._adjust_batch_size(so3_input, r3_input)

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
        train_indices = set(self.trainer.train_dataloader.dataset.selected_indices)
        val_indices = set(self.trainer.val_dataloaders.dataset.selected_indices)

        print("Training data", train_indices)
        print("Validation data", val_indices)

        if train_indices & val_indices:
            print(
                "Warning: Overlapping indices found between training and validation sets."
            )

        for prefix, dataset in [
            ("train", self.trainer.train_dataloader.dataset),
            ("val", self.trainer.val_dataloaders.dataset),
        ]:
            (
                so3_input,
                r3_input,
                norm_params,
                _,  # sdf_input
                mesh_path,
                dataset_mesh_scale,
                normalization_scale,
            ) = dataset[0]

            r3_input = denormalize_translation(r3_input, norm_params)
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

    def on_train_epoch_end(self):
        if self.current_epoch % self.config.logging.sample_every_n_epochs != 0:
            return

        (
            so3_input,
            r3_input,
            norm_params,
            _,  # sdf_input
            mesh_path,
            dataset_mesh_scale,
            normalization_scale,
        ) = self.trainer.train_dataloader.dataset[0]

        # Generate samples
        so3_output, r3_output = sample(
            self.model, r3_input.device, self.config.logging.num_samples_to_visualize
        )

        # TODO maybe make this one line (It's not a requirement.)
        r3_output = denormalize_translation(r3_output, norm_params)
        r3_input = denormalize_translation(r3_input, norm_params)

        # Compute Wasserstein distances
        w_dist_so3 = wasserstein_distance(
            so3_input.unsqueeze(0), so3_output, space="so3", method="exact", power=2
        )
        w_dist_r3 = wasserstein_distance(
            r3_input.unsqueeze(0), r3_output, space="r3", method="exact", power=2
        )

        # Log Wasserstein distances
        self.logger.experiment.log(
            {
                "val/wasserstein_distance_so3": w_dist_so3,
                "val/wasserstein_distance_r3": w_dist_r3,
            }
        )

        has_collision, scene, min_distance = check_collision_multiple_grasps(
            so3_output,
            r3_output,
            mesh_path,
            dataset_mesh_scale,
        )

        # Log each sample's visualization
        self.logger.experiment.log(
            {
                "val/generated_grasp": scene_to_wandb_image(scene),
            }
        )
