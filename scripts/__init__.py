def initialize():
    import multiprocessing
    import os

    import pytorch_lightning as pl
    import torch

    multiprocessing.set_start_method("spawn")
    os.environ["GEOMSTATS_BACKEND"] = "pytorch"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
