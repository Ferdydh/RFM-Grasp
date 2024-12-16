from src.core.config import DataConfig, MLPExperimentConfig
from src.models import pl_mlp
from src.core.train import train

if __name__ == "__main__":
    config: MLPExperimentConfig = MLPExperimentConfig.default()
    config.data = DataConfig.sanity()
    config.early_stopping.min_delta = 1e-5
    config.early_stopping.patience = 200
    config.trainer.max_epochs = 1000
    # config.data.batch_size = 1

    # Initialize model
    model = pl_mlp.SE3FMModule(config)

    train(model, config)
