from einops import rearrange
import torch
torch.set_default_dtype(torch.float64)
from so3_helpers import tangent_space_proj

class PMLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, input):
        v = self.net(input)
        x = rearrange(input[:, :-1], "b (c d) -> b c d", c=3, d=3)
        v = rearrange(v, "b (c d) -> b c d", c=3, d=3)
        Pv = tangent_space_proj(x, v)  # Pv is on the tangent space of x
        Pv = rearrange(Pv, "b c d -> b (c d)", c=3, d=3)
        return Pv
