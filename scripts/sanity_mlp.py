from src.core.config import DataConfig, MLPExperimentConfig
from src.models import pl_se3
from src.core.train import train

if __name__ == "__main__":
    config: MLPExperimentConfig = MLPExperimentConfig.default()
    config.data = DataConfig.sanity()

    config.data.sample_limit = 3  # 2 training, 1 validation
    config.data.batch_size = 2

    config.early_stopping.patience = 1000
    config.trainer.max_epochs = 10000

    # Initialize model
    model = pl_se3.FlowMatching(config)

    train(model, config)
