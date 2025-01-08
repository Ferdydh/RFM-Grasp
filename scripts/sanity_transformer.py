from src.core.config import DataConfig, TransformerExperimentConfig
from src.models import pl_se3
from src.core.train import train

from src.models.velocity_transformer import VelocityNetwork as Transformer

if __name__ == "__main__":
    config = TransformerExperimentConfig.default()
    config.data = DataConfig.sanity()
    config.data.batch_size = 64
    config.logging.sample_every_n_epochs = 100

    config.early_stopping.patience = 5000
    config.trainer.max_epochs = 5000

    # Initialize model
    model = pl_se3.FlowMatching(config, model=Transformer)

    train(model, config)
