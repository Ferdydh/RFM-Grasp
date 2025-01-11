if __name__ == "__main__":
    from scripts import initialize

    initialize()

    from src.core.config import DataConfig, ExperimentConfig
    from src.models import lightning
    from src.core.train import train

    config: ExperimentConfig = ExperimentConfig.default_mlp()
    # config.data = DataConfig.small_one_file()
    config.data = DataConfig.sanity()
    config.training.early_stop_min_delta = 1e-5
    config.training.early_stop_patience = 1000
    config.training.max_epochs = 20000

    # Initialize model
    model = lightning.Lightning(config)

    train(model, config)
