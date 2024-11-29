import torch
import torch.nn as nn
from torch import Tensor
from torchdiffeq import odeint


from .r3mlp import R3VelocityField

# from jaxtyping import Float
# from typeguard import typechecked

# TODO: Make this torch model part of R3FM configurable,
# TODO: Check Optimal Transport stuff
# TODO: Add conditioning on sdf
# TODO: Add typechecked decorator and informations


# input dim is 3 for translation always so we can remove it maybe


class R3FM(nn.Module):  
    def __init__(
        self, input_dim: int = 3, hidden_dim: int = 64, sigma_min: float = 1e-4
    ):
        #self = R3VelocityField(input_dim, hidden_dim)
        self.sigma_min = sigma_min
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # Include time t as dim+1
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hidden_dim, input_dim
            ),  # 3 for translation, we will implement SO3 later rt = expr0(tlogr0(r1)) with linalg inverse
        )

    def loss(self, x: Tensor, sigma_min: float = 1e-4) -> Tensor:
        # Question: Should we calculate one for each time step or generate one time at a time?

        t = torch.rand(x.shape[0], device=x.device).unsqueeze(-1)
        noise = torch.randn_like(x).to(x.device)

        x_t = (1 - (1 - sigma_min) * t) * noise + t * x
        optimal_flow = x - (1 - sigma_min) * noise
        predicted_flow = self(x_t, t)

        return (predicted_flow - optimal_flow).square().mean()

    # we will later use this inference function to work combined with so3 traj gen
    def inference(self, x_0: Tensor, t: Tensor, dt: Tensor) -> Tensor:

        dx_dt = self(x_0, t)
        x_t = x_0 + dt * dx_dt
        return x_t

    def generate(self, x_1: Tensor, steps: int = 100) -> Tensor:
        x_0 = torch.randn_like(x_1).to(x_1.device)
        # TODO: Implement conditioning for this generation,
        t = torch.linspace(0, 1, steps).to(x_1.device)

        def ode_func(t: Tensor, x: Tensor) -> Tensor:
            # Reshape t to match model input expectations
            t_batch = torch.full((x.shape[0], 1), t.item(), device=x_0.device)
            return self(x, t_batch)

        # TODO: Maybe we will stop using odeint after research
        trajectory = odeint(ode_func, x_0, t, method="rk4")

        return trajectory[-1]
    
    def forward(self, T: Tensor, t: Tensor) -> Tensor:
        input_data = torch.cat([T, t], dim=1)
        return self.net(input_data)
