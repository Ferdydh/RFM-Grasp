def initialize():
    import os
    import torch
    import multiprocessing
    import pytorch_lightning as pl

    multiprocessing.set_start_method("spawn")
    os.environ["GEOMSTATS_BACKEND"] = "pytorch"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    pl.seed_everything(42)

    a = 0
    print(a)
    a += 1
