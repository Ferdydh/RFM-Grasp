from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal





import torch


@dataclass
class MLPModelConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int
    sigma_min: float = 1e-4
    activation = torch.nn.ReLU
    num_hidden_layers: int = 3
    voxel_output_size: int = 256

    @classmethod
    def default(cls) -> "MLPModelConfig":
        return cls(
            input_dim=12,
            output_dim=12,
            hidden_dim=128,
        )


@dataclass
class DataConfig:
    data_path: str
    files: list[str]
    #sampler_opt: str
    batch_size: int = 32
    num_workers: int = 1
    sample_limit: Optional[int] = None
    split_ratio: float = 0.9  # Train-Val split ratio of 90-10%
    dataset_workers: int = 1
    translation_norm_param_path: Optional[str] = None
    

    @classmethod
    def sanity(cls) -> "DataConfig":
        return cls(
            data_path="data/",
            files=["Xbox360_14e5dba73b283dc7fe0939859a0b15ea_0.0005312646125977.h5"],
            batch_size=8,
            sample_limit=1,
            #sampler_opt="repeat",
        )

    @classmethod
    def small_one_file(cls) -> "DataConfig":
        return cls(
            data_path="data/",
            files=["Xbox360_14e5dba73b283dc7fe0939859a0b15ea_0.0005312646125977.h5"],
            batch_size=8,
            sample_limit=10,
            split_ratio=0.8,
            #sampler_opt="repeat",
        )
        
    @classmethod
    def two_files(cls) -> "DataConfig":
        return cls(
            data_path="data/",
            files=["Xbox360_14e5dba73b283dc7fe0939859a0b15ea_0.0005312646125977.h5",
                   "Bowl_6c3bf99a9534c75314513156cf2b8d0d_0.011015409147693228.h5"],
            batch_size=8,
            sample_limit=10,
            split_ratio=0.8,
            #sampler_opt="repeat",
        )
    

    @classmethod
    def random_h5(cls) -> "DataConfig":
        # random.seed(42)  # Fix the seed for reproducibility
        # all_h5 = list(glob("data/grasps/*.h5"))
        # random.shuffle(all_h5)
        # selected = all_h5[:100]
        return cls(
            data_path="data/",
            files=50,
            batch_size=8,
            #sampler_opt="repeat",
        )

@dataclass
class TrainingConfig:
    """Consolidated training configuration"""

    # Training parameters
    max_epochs: int = 100
    precision: Literal[16, 32, 64] = 32
    batch_accumulation: int = 1
    gradient_clip_val: float = 1.0
    r3_loss_weight: float = 3.0
    so3_loss_weight: float = 1.0
    duplicate_ratio: int = 1

    # Optimizer & Scheduler
    learning_rate: float = 1e-4
    weight_decay: float = 3e-9
    min_learning_rate: float = 1e-6
    scheduler_steps: int = 1000
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    warmup_ratio: float = 0.1

    # Validation and Logging
    validation_interval: int = 0.1
    val_every_n_epoch: int = 1
    num_samples_to_log: int = 20
    sample_interval: int = 100

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
    model: MLPModelConfig
    training: TrainingConfig

    @classmethod
    def default_mlp(cls) -> "ExperimentConfig":
        return cls(
            data=DataConfig.sanity(),
            model=MLPModelConfig.default(),
            training=TrainingConfig(),
        )
