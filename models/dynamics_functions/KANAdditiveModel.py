"""
@file MetaNeuralODE.py

Holds the model for the Neural ODE latent dynamics function
"""
import torch
import torch.nn as nn

from torchdiffeq import odeint
from models.CommonTraining import LatentDynamicsModel

from utils.layers import NumericKAN


class ODEFunction(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dynamics_net = NumericKAN(
            width=[args.in_dim + args.control_dim, args.hidden_dim, args.in_dim],
            base_fun=torch.nn.Tanh(),
            device=args.devices[0]
        )

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

        # Update the grid
        if self.n_updates < 50:
            self.dynamics_func.dynamics_net.update_grid_from_samples(torch.concatenate((x[:, 0], u), dim=-1))

        # Integrate and output
        pred = odeint(self.dynamics_func, x[:, 0], t, method='rk4')
        pred = pred.permute([1, 0, 2])
        return pred

    def model_specific_loss(self, x, domain, preds, train=True):
        """ Placeholder function for any additional loss terms a dynamics function may have """
        return 0.01 * self.dynamics_func.dynamics_net.regularization_loss()
    #
    # def model_specific_plotting(self, version_path, outputs):
    #     """ Placeholder function for any additional plots a dynamics function may have """
    #     return self.dynamics_func.dynamics_net.plot(f"{self.logger.log_dir}/signals/KAN{self.n_updates}train.png")
