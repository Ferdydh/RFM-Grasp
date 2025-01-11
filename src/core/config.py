from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union, Literal

import torch


@dataclass
class BaseModelConfig:
    """Base configuration for all models"""

    pass


@dataclass
class MLPModelConfig(BaseModelConfig):
    input_dim: int
    output_dim: int
    hidden_dim: int
    z_dim: int
    sigma_min: float = 1e-4
    activation = torch.nn.ReLU

    def __post_init__(self):
        if self.z_dim >= self.hidden_dim:
            raise ValueError("z_dim must be smaller than hidden_dim")

    @classmethod
    def default(cls) -> "MLPModelConfig":
        return cls(
            input_dim=1185,
            output_dim=1185,
            hidden_dim=1185,
            z_dim=64,
        )


@dataclass
class TransformerModelConfig(BaseModelConfig):
    sigma_min: float = 1e-4

    # TODO
    @classmethod
    def default(cls) -> "TransformerModelConfig":
        return cls()


@dataclass
class DataConfig:
    data_path: str
    files: list[str]
    sampler_opt: str
    batch_size: int = 32
    num_workers: int = 1
    sample_limit: Optional[int] = None
    split_ratio: float = 0.9  # Train-Val split ratio of 90-10%

    @classmethod
    def sanity(cls) -> "DataConfig":
        return cls(
            data_path="data/",
            files=["Xbox360_14e5dba73b283dc7fe0939859a0b15ea_0.0005312646125977.h5"],
            batch_size=8,
            sample_limit=1,
            sampler_opt="repeat",
        )

    @classmethod
    def small_one_file(cls) -> "DataConfig":
        return cls(
            data_path="data/",
            files=["Xbox360_14e5dba73b283dc7fe0939859a0b15ea_0.0005312646125977.h5"],
            batch_size=8,
            sample_limit=10,
            split_ratio=0.8,
            sampler_opt="repeat",
        )


@dataclass
class TrainingConfig:
    """Consolidated training configuration"""

    # Training parameters
    max_epochs: int = 100
    precision: Literal[16, 32, 64] = 32
    batch_accumulation: int = 1
    gradient_clip_val: float = 1.0

    # Optimizer & Scheduler
    learning_rate: float = 1e-4
    weight_decay: float = 3e-9
    min_learning_rate: float = 1e-6
    scheduler_steps: int = 1000

    # Validation and Logging
    validation_interval: int = 5
    log_interval: int = 50
    num_samples_to_log: int = 2
    sample_interval: int = 50

    # Checkpointing
    checkpoint_dir: str = "logs/checkpoints"
    checkpoint_name: str = "model-{epoch:02d}-{val_loss:.2f}"
    checkpoint_metric: str = "val/loss"
    checkpoint_mode: Literal["min", "max"] = "min"
    keep_top_k_checkpoints: int = 3
    save_last: bool = True

    # Early Stopping
    early_stop_patience: int = 200
    early_stop_min_delta: float = 1e-5

    # Project Metadata
    project_name: str = "adlr"
    run_name: Optional[str] = None
    save_dir: str = "logs"

    def __post_init__(self):
        if self.run_name is None:
            self.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


@dataclass
class ExperimentConfig:
    """Unified experiment configuration that works with any model type"""

    data: DataConfig
    model: Union[MLPModelConfig, TransformerModelConfig]
    training: TrainingConfig

    @classmethod
    def default_mlp(cls) -> "ExperimentConfig":
        return cls(
            data=DataConfig.sanity(),
            model=MLPModelConfig.default(),
            training=TrainingConfig(),
        )

    @classmethod
    def default_transformer(cls) -> "ExperimentConfig":
        return cls(
            data=DataConfig.sanity(),
            model=TransformerModelConfig.default(),
            training=TrainingConfig(),
        )
