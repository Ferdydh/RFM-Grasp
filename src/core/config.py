from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import torch


# Base Configs
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
    num_workers: int = 4
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

    # TODO: Add more data presets


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-5
    eps: float = 1e-8

    @classmethod
    def default(cls) -> "OptimizerConfig":
        return cls(
            name="adamw",
            lr=1e-4,
            weight_decay=3e-9,
        )


@dataclass
class SchedulerConfig:
    name: str
    T_max: Optional[int] = None
    eta_min: Optional[float] = None

    @classmethod
    def cosine_default(cls) -> "SchedulerConfig":
        return cls(
            name="cosine",
            T_max=1000,
            eta_min=1e-6,
        )


@dataclass
class TrainerConfig:
    max_epochs: int
    precision: Literal[16, 32, 64]
    gradient_clip_val: float
    accumulate_grad_batches: int
    check_val_every_n_epoch: int
    log_every_n_steps: int

    @classmethod
    def sanity(cls) -> "TrainerConfig":
        return cls(
            max_epochs=10,
            precision=64,
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            check_val_every_n_epoch=5,
            log_every_n_steps=1,
        )

    @classmethod
    def default(cls) -> "TrainerConfig":
        return cls(
            max_epochs=100,
            precision=64,
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            check_val_every_n_epoch=5,
            log_every_n_steps=50,
        )


@dataclass
class LoggingConfig:
    project_name: str
    save_dir: str
    num_samples_to_visualize: int
    sample_every_n_epochs: int
    log_every_n_steps: int
    run_name: Optional[str] = None
    log_model: bool = False

    @classmethod
    def sanity(cls) -> "LoggingConfig":
        return cls(
            project_name="adlr",
            save_dir="logs",
            num_samples_to_visualize=1,
            sample_every_n_epochs=1,
            log_every_n_steps=1,
        )

    @classmethod
    def default(cls) -> "LoggingConfig":
        return cls(
            project_name="adlr",
            save_dir="logs",
            num_samples_to_visualize=3,
            sample_every_n_epochs=50,
            log_every_n_steps=50,
        )


@dataclass
class CheckpointConfig:
    dirpath: str
    filename: str
    monitor: str
    mode: Literal["min", "max"]
    save_last: bool
    save_top_k: int

    @classmethod
    def default(cls) -> "CheckpointConfig":
        return cls(
            dirpath="checkpoints",
            filename="fm-{epoch:02d}-{val_loss:.2f}",
            monitor="val/loss",
            mode="min",
            save_last=True,
            save_top_k=3,
        )


@dataclass
class EarlyStoppingConfig:
    monitor: str
    min_delta: float
    patience: int
    mode: Literal["min", "max"]

    @classmethod
    def sanity(cls) -> "EarlyStoppingConfig":
        return cls(
            monitor="val/loss",
            min_delta=1e-4,
            patience=5,
            mode="min",
        )

    @classmethod
    def default(cls) -> "EarlyStoppingConfig":
        return cls(
            monitor="val/loss",
            min_delta=1e-5,
            patience=200,
            mode="min",
        )


@dataclass
class TrainingConfig:
    """Contrastive learning configuration for transformer"""

    gamma: float
    reduction: str
    temperature: float
    contrast: str
    z_var_penalty: float

    @classmethod
    def default(cls) -> "TrainingConfig":
        return cls(
            gamma=0.05,
            reduction="mean",
            temperature=0.1,
            contrast="simclr",
            z_var_penalty=0.0,
        )


@dataclass
class BaseExperimentConfig:
    """Base configuration for all experiments"""

    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    trainer: TrainerConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    early_stopping: EarlyStoppingConfig


@dataclass
class MLPExperimentConfig(BaseExperimentConfig):
    model: MLPModelConfig

    @classmethod
    def default(cls) -> "MLPExperimentConfig":
        return cls(
            data=DataConfig.sanity(),
            model=MLPModelConfig.default(),
            optimizer=OptimizerConfig.default(),
            scheduler=SchedulerConfig.cosine_default(),
            trainer=TrainerConfig.sanity(),
            logging=LoggingConfig.sanity(),
            checkpoint=CheckpointConfig.default(),
            early_stopping=EarlyStoppingConfig.default(),
        )


@dataclass
class TransformerExperimentConfig(BaseExperimentConfig):
    model: TransformerModelConfig
    training: TrainingConfig

    @classmethod
    def default(cls) -> "TransformerExperimentConfig":
        return cls(
            data=DataConfig.sanity(),
            model=TransformerModelConfig.default(),
            training=TrainingConfig.default(),
            optimizer=OptimizerConfig.default(),
            scheduler=SchedulerConfig.cosine_default(),
            trainer=TrainerConfig.sanity(),
            logging=LoggingConfig.sanity(),
            checkpoint=CheckpointConfig.default(),
            early_stopping=EarlyStoppingConfig.default(),
        )
