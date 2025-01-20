if __name__ == "__main__":
    from scripts import initialize

    initialize()

    from src.core.config import DataConfig, ExperimentConfig
    from src.models import lightning
    from src.core.train import train

    config: ExperimentConfig = ExperimentConfig.default_mlp()
    config.data = DataConfig.sanity()
    config.data.split_ratio = 0.9

    # Sanity with 1
    # config.data.sample_limit = 1
    # config.data.batch_size = 1

    # Check if it can learn multiple samples
    # Starting with 4
    config.data.sample_limit = 5
    config.data.batch_size = 4
    config.data.num_workers = 3

    # Model
    # config.model.hidden_dim = 512
    config.model.hidden_dim = 1024
    config.training.early_stop_patience = 100
    config.training.max_epochs = 20000

    config.training.sample_interval = 1000

    # Initialize model
    model = lightning.Lightning(config)

    train(model, config)
