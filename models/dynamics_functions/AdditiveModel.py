"""
@file MetaNeuralODE.py

Holds the model for the Neural ODE latent dynamics function
"""
import torch
import torch.nn as nn

from torchdiffeq import odeint
from models.CommonTraining import LatentDynamicsModel


class ODEFunction(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dynamics_net = nn.Sequential(
            nn.Linear(args.in_dim + args.control_dim, args.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(args.hidden_dim, args.in_dim),
        )

        nn.init.normal_(self.dynamics_net[0].weight, 0, 0.01)
        nn.init.zeros_(self.dynamics_net[0].bias)

        nn.init.normal_(self.dynamics_net[-1].weight, 0, 0.01)
        nn.init.zeros_(self.dynamics_net[-1].bias)

    def set_control(self, control):
        self.controls = control

    def forward(self, t, x):
        x = torch.concatenate((x, self.controls), dim=-1)
        return self.dynamics_net(x)


class AdditiveModel(LatentDynamicsModel):
    def __init__(self, args):
        """ Latent dynamics as parameterized by a global deterministic neural ODE """
        super().__init__(args)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODEFunction(args)

    def forward(self, x, u, generation_len):
        """ Forward function of the ODE network """
        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(1, self.args.timesteps, self.args.timesteps, device='cuda')

        # Set controls for this batch
        self.dynamics_func.set_control(u)

        # Integrate and output
        pred = odeint(self.dynamics_func, x[:, 0], t)
        pred = pred.permute([1, 0, 2])
        return pred