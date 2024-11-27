import torch
from .so3_helpers import norm_SO3


def loss_fn(v, u, x):
    res = v - u
    norm = norm_SO3(x, res)  # norm-squared on SO(3)
    loss = torch.mean(norm, dim=-1)
    return loss
