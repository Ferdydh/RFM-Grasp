import torch
import torch.nn as nn
from torch import Tensor
from torchdiffeq import odeint

class R3FM(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, sigma_min: float = 1e-4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sigma_min = sigma_min
        
        # Velocity field network
        self.velocity_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # Include time t as dim+1
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass of the velocity field network.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim]
            t (Tensor): Time tensor of shape [batch_size, 1]
            
        Returns:
            Tensor: Predicted velocity field
        """
        input_data = torch.cat([x, t], dim=1)
        return self.velocity_net(input_data)

    def loss(self, x: Tensor) -> Tensor:
        """Compute the loss for training.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tensor: Scalar loss value
        """
        # Sample random time steps
        t = torch.rand(x.shape[0], device=x.device).unsqueeze(-1)
        noise = torch.randn_like(x).to(x.device)

        # Compute noisy sample at time t
        x_t = (1 - (1 - self.sigma_min) * t) * noise + t * x
        
        # Compute optimal flow
        optimal_flow = x - (1 - self.sigma_min) * noise
        
        # Get predicted flow
        predicted_flow = self.forward(x_t, t)

        # Compute MSE loss
        return (predicted_flow - optimal_flow).square().mean()

    def inference(self, x_0: Tensor, t: Tensor, dt: Tensor) -> Tensor:
        """Single step inference.
        
        Args:
            x_0 (Tensor): Initial state tensor
            t (Tensor): Current time
            dt (Tensor): Time step size
            
        Returns:
            Tensor: Next state prediction
        """
        dx_dt = self.forward(x_0, t)
        x_t = x_0 + dt * dx_dt
        return x_t

    def generate(self, x_1: Tensor, steps: int = 100) -> Tensor:
        """Generate a complete trajectory.
        
        Args:
            x_1 (Tensor): Target state tensor
            steps (int, optional): Number of integration steps. Defaults to 100.
            
        Returns:
            Tensor: Generated final state
        """
        x_0 = torch.randn_like(x_1).to(x_1.device)
        t = torch.linspace(0, 1, steps).to(x_1.device)

        def ode_func(t: float, x: Tensor) -> Tensor:
            t_batch = torch.full((x.shape[0], 1), t.item(), device=x_0.device)
            return self.forward(x, t_batch)

        trajectory = odeint(ode_func, x_0, t, method="rk4")
        return trajectory[-1]