from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor

from src.core.config import MLPExperimentConfig

from .r3fm import R3FM
from .so3fm import SO3FM

from .wasserstein import wasserstein_distance


class SE3FMModule(pl.LightningModule):
    def __init__(self, config: MLPExperimentConfig):
        super().__init__()
        # There are some loss issues related to quaternions
        # that we need to solve for float32

        # torch.set_float32_matmul_precision("medium")
        # torch.set_default_dtype(torch.float32)
        self.config = config

        self.so3fm = SO3FM()
        self.r3fm = R3FM()

    def compute_loss(
        self,
        so3_inputs,
        r3_inputs,
        prefix: str = "train",
    ) -> Tuple[Tensor, dict[str, Tensor]]:
        so3_loss = self.so3fm.loss(so3_inputs)
        r3_loss = self.r3fm.loss(r3_inputs)
        loss = so3_loss + r3_loss
        return loss, {f"{prefix}/so3loss": loss}

    def forward(self, so3_input, r3_input):
        so3_output = self.so3fm.forward(so3_input)
        r3_output = self.r3fm.forward(r3_input)
        return so3_output, r3_output

    def training_step(self, batch, batch_idx):  # We will add conditioning here
        (
            so3_input,
            r3_input,
            sdf_input,
            mesh_path,
            dataset_mesh_scale,
            normalization_scale,
        ) = batch

        # TODO: Implement log dict here
        loss, log_dict = self.compute_loss(so3_input, r3_input, "train")
        if batch_idx % 100 == 0:
            self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (
            so3_input,
            r3_input,
            sdf_input,
            mesh_path,
            dataset_mesh_scale,
            normalization_scale,
        ) = batch

        # These generate function later only take sdf input
        # Loss here is does care about generation,
        # it is not about time t velocity estimation

        r3_generated = self.r3fm.generate(r3_input)

        so3_generated = self.so3fm.generate(so3_input).unsqueeze(0)

        print("R3 Ground Truth:", r3_input, "\nR3 Generated:", r3_generated)
        print("SO3 Input:", so3_input, "\nSO3 Generated:", so3_generated)

        # Calculate Wasserstein distance the calculation
        # at the top won't be used if we use this

        # r3_wasserstein = wasserstein_distance(
        #     r3_generated, r3_input, space="r3", method="exact", power=2
        # )
        # so3_wasserstein = wasserstein_distance(
        #     so3_generated, so3_input, space="so3", method="exact", power=2
        # )

        ## Calculate total validation loss
        # val_loss = r3_wasserstein + so3_wasserstein

        ## Log metrics with val/ prefix
        # self.log('val/loss', val_loss, prog_bar=True)
        # self.log('val/r3_wasserstein', r3_wasserstein)
        # self.log('val/so3_wasserstein', so3_wasserstein)

        val_loss = torch.mean(torch.abs(r3_generated - r3_input)) + torch.mean(
            torch.abs(so3_generated - so3_input)
        )

        self.log("val/loss", val_loss, prog_bar=True)

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
