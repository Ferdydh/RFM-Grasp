from geomstats._backend import _backend_config as _config
import torch
import warnings
import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

warnings.filterwarnings("ignore")

_config.DEFAULT_DTYPE = torch.cuda.FloatTensor
