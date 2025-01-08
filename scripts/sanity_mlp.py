from src.core.config import DataConfig, ExperimentConfig
from src.models import lightning
from src.core.train import train

if __name__ == "__main__":
    config: ExperimentConfig = ExperimentConfig.default_mlp()
    config.data = DataConfig.sanity()

    # config.data.sample_limit = 3  # 2 training, 1 validation
    config.data.sample_limit = 1  # overfitting
    config.data.batch_size = 512

    config.training.early_stop_patience = 100
    config.training.max_epochs = 1000

    config.training.sample_interval = 25

    # Initialize model
    model = lightning.Lightning(config)

    train(model, config)
