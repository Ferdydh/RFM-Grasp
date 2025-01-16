if __name__ == "__main__":
    import torch
    print(torch.cuda.is_available())  # Should return True if GPUs are accessible
    print(torch.version.cuda)   
    from scripts import initialize

    initialize()

    from src.core.config import DataConfig, ExperimentConfig
    from src.models import lightning
    from src.core.train import train

    config: ExperimentConfig = ExperimentConfig.default_mlp()
    config.data = DataConfig.sanity()
    config.data.split_ratio = 1.0  # No validation set
    print(config.data.split_ratio)
    # config.data.sample_limit = 3  # 2 training, 1 validation
    config.data.sample_limit = 100  # overfitting
    config.data.batch_size = 8192

    config.training.early_stop_patience = 100
    config.training.max_epochs = 20000

    config.training.sample_interval = 1000

    # Initialize model
    model = lightning.Lightning(config)

    train(model, config)
