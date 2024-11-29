
import torch
from einops import rearrange
from geomstats._backend import _backend_config as _config
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from scipy.spatial.transform import Rotation
import torch.nn as nn

from .pmlp import PMLP
from .so3_condflowmatcher import SO3ConditionalFlowMatcher
from .so3_helpers import expmap, loss_so3

_config.DEFAULT_DTYPE = torch.cuda.FloatTensor

# TODO: Implement conditioning


class SO3FM(PMLP):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):

        self.so3_group = SpecialOrthogonal(n=3, point_type="matrix")
        self.so3_cfm = SO3ConditionalFlowMatcher(manifold=self.so3_group)
        self.model = PMLP(9, time_varying=True)

    def inference(self, xt, t, dt):
        with torch.no_grad():
            vt = self.model(
                torch.cat([xt, t[:, None]], dim=-1)
            )  # vt on the tanget of xt
            vt = rearrange(vt, "b (c d) -> b c d", c=3, d=3)
            xt = rearrange(xt, "b (c d) -> b c d", c=3, d=3)
            xt_new = expmap(xt, vt * dt)  # expmap to get the next point
        return rearrange(xt_new, "b c d -> b (c d)", c=3, d=3)

    def generate(self, x: torch.Tensor, steps=100):  # x batch of sdfs
        n_test = x.shape[0]
        traj = torch.tensor(Rotation.random(n_test).as_matrix()).reshape(-1, 9)
        for t in torch.linspace(0, 1, steps):
            t = torch.tensor([t], dtype=torch.float64).repeat(n_test)
            dt = torch.tensor([1 / steps])
            traj = self.inference(traj, t, dt)
        final_traj = rearrange(traj, "b (c d) -> b c d", c=3, d=3)
        return final_traj[-1]

    # we can add device as an argument but if we put x1
    # to device then we can leave it as
    def loss(self, x: torch.Tensor):  # add additional arg for sdf here
        x1 = x.double()
        x0 = torch.tensor(
            Rotation.random(x1.size(0)).as_matrix(), dtype=torch.float64
        ).to(x1.device)

        t, xt, ut = self.so3_cfm.sample_location_and_conditional_flow_simple(x0, x1)

        vt = self.model(
            torch.cat([rearrange(xt, "b c d -> b (c d)", c=3, d=3), t[:, None]], dim=-1)
        )
        vt = rearrange(vt, "b (c d) -> b c d", c=3, d=3)

        loss = loss_so3(vt, ut, xt)

        return loss
