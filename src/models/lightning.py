from models.util import duplicate_batch_to_size
import torch.nn.functional as F
from einops import rearrange
from scipy.spatial.transform import Rotation
from typing import Tuple, Dict
import pytorch_lightning as pl
import torch
from torch import Tensor
import wandb
from torch.utils.data import DataLoader


from src.data.util import denormalize_translation
from src.core.config import ExperimentConfig
from src.core.visualize import (
    check_collision,
    check_collision_multiple_grasps,
    scene_to_wandb_3d,
)
from src.models.flow import sample, sample_location_and_conditional_flow
from src.models.velocity_mlp import VelocityNetwork
from src.models.wasserstein import wasserstein_distance


class Lightning(pl.LightningModule):
    """Flow Matching model combining SO3 and R3 manifold learning with synchronized time sampling."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.model = VelocityNetwork(self.config)

        # TODO use config
        self.save_hyperparameters()

    def compute_loss(
        self,
        so3_inputs: Tensor,
        r3_inputs: Tensor,
        sdf_inputs: Tensor,
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

        # SO3 computation - already in [batch, 3, 3] format
        x0_so3 = torch.tensor(
            Rotation.random(so3_inputs.size(0)).as_matrix(), device=so3_inputs.device
        )  # Shape: [batch, 3, 3]

        # Sample location and flow for SO3
        xt_so3, ut_so3 = sample_location_and_conditional_flow(x0_so3, so3_inputs, t)
        # Both xt_so3 and ut_so3 are [batch, 3, 3]

        # R3 computation with same time points
        t_expanded = t.unsqueeze(-1)  # [batch, 1]
        noise = torch.randn_like(r3_inputs)

        # Get predicted flow for R3
        x_t_r3 = (
            1 - (1 - self.config.model.sigma_min) * t_expanded
        ) * noise + t_expanded * r3_inputs

        # Forward pass now expects [batch, 3, 3] format
        vt_so3, predicted_flow = self.model.forward(
            xt_so3, x_t_r3, sdf_inputs, t_expanded
        )
        # vt_so3 is now directly [batch, 3, 3]

        # Compute SO3 loss using Riemannian metric
        r = torch.transpose(xt_so3, dim0=-2, dim1=-1) @ (vt_so3 - ut_so3)
        norm = -torch.diagonal(r @ r, dim1=-2, dim2=-1).sum(dim=-1) / 2
        so3_loss = torch.mean(norm, dim=-1)

        # Compute noisy sample and optimal flow for R3
        optimal_flow = r3_inputs - (1 - self.config.model.sigma_min) * noise
        r3_loss = F.mse_loss(predicted_flow, optimal_flow)

        # Works better in this setup but we can change later
        total_loss = (
            self.config.training.so3_loss_weight * so3_loss
            + self.config.training.r3_loss_weight * r3_loss
        )

        loss_dict = {
            f"{prefix}/so3_loss": so3_loss,
            f"{prefix}/r3_loss": r3_loss,
            f"{prefix}/loss": total_loss,
        }

        return total_loss, loss_dict

    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        so3_input, r3_input, sdf_input, *_ = duplicate_batch_to_size(batch)
        loss, log_dict = self.compute_loss(so3_input, r3_input, sdf_input, "train")

        self.log_dict(
            log_dict,
            prog_bar=True,
            batch_size=self.config.data.batch_size,
        )

        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, Tensor]:
        so3_input, r3_input, sdf_input, *_ = batch

        with torch.enable_grad():
            loss, log_dict = self.compute_loss(so3_input, r3_input, sdf_input, "val")

        # Log validation metrics
        self.log_dict(
            log_dict,
            prog_bar=True,
            batch_size=self.config.data.batch_size,
        )

        return log_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            betas=tuple(self.config.training.adamw_betas),
            eps=self.config.training.epsilon,
            weight_decay=self.config.training.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches

        # Single linear scheduler with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.training.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.training.warmup_ratio,
            anneal_strategy="linear",
            div_factor=3.0,  # initial_lr = max_lr/div_factor
            final_div_factor=float("inf"),  # final_lr = initial_lr/final_div_factor
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": self.config.training.checkpoint_metric,
            },
        }

    def on_train_start(self) -> None:
        """Setup logging of initial grasp scenes on training start."""
        train_dataset = self.trainer.train_dataloader.dataset
        val_dataset = self.trainer.val_dataloaders.dataset

        # Get base datasets (handle Subset case)
        train_base = (
            train_dataset.dataset
            if isinstance(train_dataset, torch.utils.data.Subset)
            else train_dataset
        )
        val_base = (
            val_dataset.dataset
            if isinstance(val_dataset, torch.utils.data.Subset)
            else val_dataset
        )

        # First get selected_indices from the base dataset if they exist
        base_selected = (
            train_base.selected_indices
            if hasattr(train_base, "selected_indices")
            else None
        )

        # Then get the actual split indices from Subset
        if isinstance(train_dataset, torch.utils.data.Subset):
            train_indices = set(
                train_dataset.indices
            )  # These are indices into the base dataset
            val_indices = set(val_dataset.indices)

            # If base dataset had selected_indices, we need to map through them
            if base_selected is not None:
                train_indices = set(base_selected[i] for i in train_indices)
                val_indices = set(base_selected[i] for i in val_indices)
        else:
            # If not a subset, use selected_indices directly if they exist
            train_indices = (
                set(train_base.selected_indices)
                if hasattr(train_base, "selected_indices")
                else None
            )
            val_indices = (
                set(val_base.selected_indices)
                if hasattr(val_base, "selected_indices")
                else None
            )

        if train_indices is not None and val_indices is not None:
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
                sdf_input,  # sdf_input
                norm_params,
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
                    f"{prefix}/original_grasp": scene_to_wandb_3d(scene),
                    # f"{prefix}/original_grasp_transform": gripper_transform,
                }
            )

    def on_train_epoch_end(self):
        if self.current_epoch % self.config.training.sample_interval != 0:
            return
        random_idx = torch.randint(0, len(self.trainer.train_dataloader.dataset), (1,))
        (
            so3_input,
            r3_input,
            sdf_input,  # sdf_input
            norm_params,
            mesh_path,
            dataset_mesh_scale,
            normalization_scale,
        ) = self.trainer.train_dataloader.dataset[random_idx]
        # print('Initialtype',type(sdf_input))
        # Generate samples
        sdf_input = rearrange(sdf_input, "... -> 1 1 ...")

        # print(r3_input.shape)
        so3_output, r3_output = sample(
            self.model,
            sdf_input,
            r3_input.device,
            self.config.training.num_samples_to_log,
        )

        # TODO maybe make this one line (It's not a requirement.)
        r3_output = denormalize_translation(r3_output, norm_params)
        r3_input = denormalize_translation(r3_input, norm_params)

        # # Compute Wasserstein distances
        # w_dist_so3 = wasserstein_distance(
        #     so3_input.unsqueeze(0), so3_output, space="so3", method="exact", power=2
        # )
        # w_dist_r3 = wasserstein_distance(
        #     r3_input.unsqueeze(0), r3_output, space="r3", method="exact", power=2
        # )

        # Log Wasserstein distances
        # self.logger.experiment.log(
        #     {
        #         "val/wasserstein_distance_so3": w_dist_so3,
        #         "val/wasserstein_distance_r3": w_dist_r3,
        #     }
        # )

        has_collision, scene, min_distance = check_collision_multiple_grasps(
            so3_output,
            r3_output,
            mesh_path,
            dataset_mesh_scale,
        )

        # Log each sample's visualization
        self.logger.experiment.log(
            {
                "val/generated_grasp": scene_to_wandb_3d(scene),
            }
        )
