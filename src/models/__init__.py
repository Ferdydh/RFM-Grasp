import os
# This line magically changes some tensors to double precision
# so we need to reset the default dtype later.

os.environ["GEOMSTATS_BACKEND"] = "pytorch"


from geomstats._backend import _backend_config as _config
import torch
import warnings

# warnings.filterwarnings("ignore")
# _config.DEFAULT_DTYPE = torch.cuda.FloatTensor
