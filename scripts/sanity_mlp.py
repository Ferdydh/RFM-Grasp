from src.core.config import DataConfig, MLPExperimentConfig
from src.models import pl_se3
from src.core.train import train

if __name__ == "__main__":
    config: MLPExperimentConfig = MLPExperimentConfig.default()
    config.data = DataConfig.sanity()

    # config.data.sample_limit = 3  # 2 training, 1 validation
    config.data.sample_limit = 1  # overfitting
    config.data.batch_size = 512

    config.early_stopping.patience = 100
    config.trainer.max_epochs = 1000

    config.logging.sample_every_n_epochs = 25

    # Initialize model
    model = pl_se3.FlowMatching(config)

    train(model, config)
