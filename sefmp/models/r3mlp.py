import torch
import torch.nn as nn
from torch import Tensor

# TODO: Make this torch model part of R3FM configurable,
# TODO: Check Optimal Transport stuff
# TODO: Add conditioning on sdf
# TODO: Add typechecked decorator and informations

# input dim is 3 for translation always so we can remove it maybe


class R3VelocityField(nn.Module):
    def __init__(
        self, input_dim: int = 3, hidden_dim: int = 64
    ):  # trial for translation
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # Include time t as dim+1
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hidden_dim, input_dim
            ),  # 3 for translation, we will implement SO3 later rt = expr0(tlogr0(r1)) with linalg inverse
        )

    def forward(self, T: Tensor, t: Tensor) -> Tensor:
        input_data = torch.cat([T, t], dim=1)
        return self.net(input_data)
