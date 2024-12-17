from src.core.config import DataConfig, MLPExperimentConfig
from src.models import pl_se3
from src.core.train import train

if __name__ == "__main__":
    config: MLPExperimentConfig = MLPExperimentConfig.default()
    # config.data = DataConfig.small_one_file()
    config.data = DataConfig.sanity()
    config.early_stopping.min_delta = 1e-5
    config.early_stopping.patience = 3
    config.trainer.max_epochs = 100
    # config.data.sample_limit = 1
    # config.data.batch_size = 1

    # Initialize model
    model = pl_se3.FlowMatching(config)

    train(model, config)
