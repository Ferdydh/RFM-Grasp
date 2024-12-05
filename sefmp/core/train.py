import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

import wandb
from datetime import datetime
import atexit

from .utils import load_config, get_device
from data.grasp_dataset import GraspDataModule
from models.se3lightning import SE3FMModule


def cleanup_wandb():
    """Ensure wandb run is properly closed."""
    try:
        wandb.finish()
    except:
        pass


def train(experiment: str = "sanity_check"):
    """Train the model with lightning."""

    # pass

    # Register cleanup function
    atexit.register(cleanup_wandb)

    try:
        # Load configuration
        cfg = load_config(experiment)

        # TODO: Utilize device
        device = get_device()

        # Setup unique run name if not specified
        if cfg["logging"]["run_name"] is None:
            cfg["logging"]["run_name"] = (
                f"sefmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        # Initialize WandB logger with modified settings
        wandb_logger = WandbLogger(
            project=cfg["logging"]["project_name"],
            name=cfg["logging"]["run_name"],
            log_model=cfg["logging"]["log_model"],
            save_dir=cfg["logging"]["save_dir"],
            settings=wandb.Settings(start_method="thread"),
        )

        # Log hyperparameters and config file
        wandb_logger.log_hyperparams(cfg)
        wandb.save(experiment)  # Save the original config file as an artifact

        # Initialize DataModule with multiple selectors
        grasp_data = GraspDataModule(
            data_root=cfg["data"]["data_path"],
            grasp_files=cfg["data"]["grasp_files"],  # Using list of selectors
            sampler_opt=cfg["data"]["sampler_opt"],
            batch_size=cfg["data"]["batch_size"],
            num_samples=1,  # Optional: limit total samples
            num_workers=cfg["data"]["num_workers"],
        )

        # Set up the data module
        grasp_data.setup()

        # Get data loaders
        train_loader = grasp_data.train_dataloader()
        val_loader = grasp_data.val_dataloader()

        # Setup callbacks
        callbacks = []

        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg["checkpoint"]["dirpath"],
            filename=cfg["checkpoint"]["filename"],
            monitor=cfg["checkpoint"]["monitor"],
            mode=cfg["checkpoint"]["mode"],
            save_last=cfg["checkpoint"]["save_last"],
            save_top_k=cfg["checkpoint"]["save_top_k"],
        )
        callbacks.append(checkpoint_callback)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor=cfg["early_stopping"]["monitor"],
            min_delta=cfg["early_stopping"]["min_delta"],
            patience=cfg["early_stopping"]["patience"],
            verbose=True,
            mode=cfg["early_stopping"]["mode"],
            check_finite=True,  # Stop if loss becomes NaN or inf
            stopping_threshold=cfg["early_stopping"].get(
                "stopping_threshold", None
            ),  # Optional absolute threshold
            divergence_threshold=cfg["early_stopping"].get(
                "divergence_threshold", None
            ),  # Optional divergence threshold
        )
        callbacks.append(early_stopping)

        # Initialize trainer
        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=callbacks,
            max_epochs=cfg["trainer"]["max_epochs"],
            accelerator="cpu",  # Change this to explicitly use CPU
            devices=1,  # Use single CPU device
            precision=cfg["trainer"][
                "precision"
            ],  # Currently we force 64-bit precision
            gradient_clip_val=cfg["trainer"]["gradient_clip_val"],
            accumulate_grad_batches=cfg["trainer"]["accumulate_grad_batches"],
            # val_check_interval=cfg["trainer"]["val_check_interval"], we check every n epochs currently
            check_val_every_n_epoch=cfg["trainer"]["check_val_every_n_epoch"],
            log_every_n_steps=cfg["trainer"]["log_every_n_steps"],
        )

        # Add this right before trainer.fit()
        wandb.require("service")

        # Initialize model
        model = SE3FMModule(cfg)

        # Train model
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        raise

    finally:
        # Ensure wandb is properly closed
        cleanup_wandb()
        print("\nWandB run closed. Exiting...")


if __name__ == "__main__":
    train()
