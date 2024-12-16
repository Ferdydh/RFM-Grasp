from dataclasses import asdict
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

from src.data.grasp_dataset import DataHandler

from src.core.config import (
    MLPExperimentConfig,
    TransformerExperimentConfig,
)


def cleanup_wandb():
    """Ensure wandb run is properly closed."""
    try:
        wandb.finish()
    except:
        pass


def train(
    model: pl.LightningModule,
    config: MLPExperimentConfig | TransformerExperimentConfig,
):
    """Train the autoencoder model with improved logging and visualization."""
    # Register cleanup function
    atexit.register(cleanup_wandb)

    try:
        # Load configuration
        # Setup unique run name if not specified
        if config.logging.run_name is None:
            config.logging.run_name = f"fm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize WandB logger with modified settings
        wandb_logger = WandbLogger(
            project=config.logging.project_name,
            name=config.logging.run_name,
            log_model=config.logging.log_model,
            save_dir=config.logging.save_dir,
            settings=wandb.Settings(start_method="thread"),
        )

        # Log hyperparameters and config file
        wandb_logger.log_hyperparams(asdict(config))

        # Initialize data handler
        data_handler = DataHandler(config)

        # Get dataloaders
        # train_loader = data_handler.train_dataloader()
        # val_loader = data_handler.val_dataloader()

        # Setup callbacks
        callbacks = []

        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.checkpoint.dirpath,
            filename=config.checkpoint.filename,
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_last=config.checkpoint.save_last,
            save_top_k=config.checkpoint.save_top_k,
        )
        callbacks.append(checkpoint_callback)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # Early stopping callback
        # early_stopping = EarlyStopping(
        #     monitor=config.early_stopping.monitor,
        #     min_delta=config.early_stopping.min_delta,
        #     patience=config.early_stopping.patience,
        #     verbose=True,
        #     mode=config.early_stopping.mode,
        #     check_finite=True,  # Stop if loss becomes NaN or inf
        # )
        # callbacks.append(early_stopping)

        # Initialize trainer
        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=callbacks,
            max_epochs=config.trainer.max_epochs,
            # accelerator="auto",
            # devices="auto",
            accelerator="cpu",
            devices=1,
            # TODO: fix this, try to use float32 instead
            precision=config.trainer.precision,
            gradient_clip_val=config.trainer.gradient_clip_val,
            accumulate_grad_batches=config.trainer.accumulate_grad_batches,
            val_check_interval=1.0,
            log_every_n_steps=config.trainer.log_every_n_steps,
        )

        # Add this right before trainer.fit()
        wandb.require("service")

        # Train model
        trainer.fit(model=model, datamodule=data_handler)

        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        raise

    finally:
        # Ensure wandb is properly closed
        cleanup_wandb()
        print("\nWandB run closed. Exiting...")
