import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.integrate import odeint


# X0 and Integrate Window
x0 = np.array([10])
t = np.linspace(-1, 0, 1000)
num_samples = 100

# Additive Function
def f_add(x, t, u):
    return np.sin(x) + u


plot = False
if plot:
    avgs = []
    for x_i in tqdm(np.linspace(1, 99, num_samples)):
        for u_i in reversed(np.linspace(-5, 5, num_samples)):
            sol = odeint(f_add, x_i, t, args=(np.array([u_i]),))
            plt.plot(t, sol)
            avgs.append(np.mean(sol))
    plt.title("Dataset of sin(x) + u")
    plt.show()
    plt.close()

    avgs = []
    for u_i in reversed(np.linspace(-2, 2, num_samples)):
        sol = odeint(f_add, x0, t, args=(np.array([u_i]),))
        plt.plot(t, sol)
        avgs.append(np.mean(sol))
    plt.title("Single x_i example of u influence on sin(x) + u")
    plt.show()
    plt.close()

    plt.plot(np.linspace(1, 99, num_samples), avgs)
    plt.title("Additive: Avg Signal Value vs Value of U")
    plt.show()
    plt.close()


# Multiplicative Function
def f_multi(x, t, u):
    return u * np.sin(x)

plot = True
if plot:
    # avgs = []
    # for x_i in tqdm(np.linspace(1, 99, num_samples)):
    #     for u_i in reversed(np.linspace(-2, 2, num_samples)):
    #         sol = odeint(f_multi, x_i, t, args=(np.array([u_i]),))
    #         plt.plot(t, sol)
    #         avgs.append(np.mean(sol))
    # plt.title("Dataset of sin(x) * u")
    # plt.show()
    # plt.close()

    avgs = []
    for u_i in np.linspace(-5, 5, num_samples):
        sol = odeint(f_multi, x0, t, args=(np.array([u_i]),))
        plt.plot(t, sol)
        avgs.append(np.mean(sol))
    plt.title("Single x_i example of u influence on sin(x) * u")
    plt.show()
    plt.close()

    # plt.plot(np.linspace(1, 99, num_samples), avgs)

    plt.plot(range(20), avgs[:20], c='k')
    plt.plot(range(20, 60), avgs[20:60], c='b')
    plt.plot(range(60, 100), avgs[60:100], c='orange')

    plt.title("Multiplicative: Avg Signal Value vs Value of U")
    plt.show()
    plt.close()


""" MLP to learn function """
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint as odeint_torch


class StandardODE(nn.Module):
    def __init__(self, in_dim=1, out_dim=1):
        super().__init__()
        self.dynamics_net = nn.Sequential(
            nn.Linear(in_dim, 4),
            nn.LeakyReLU(),
            nn.Linear(4, out_dim)
        )

    def set_us(self, u):
        self.u = u

    def forward(self, t, x):
        x = torch.concatenate((x, self.u), dim=-1)

        return self.dynamics_net(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ode = StandardODE(in_dim=2)

    def forward(self, x):
        t = torch.linspace(1, 100, 100, device='cuda')

        pred = odeint_torch(self.ode, x, t)
        pred = pred.permute([1, 0, 2])
        return pred


net = Net()
mse = nn.MSELoss()
optim = torch.optim.AdamW(net.parameters(), lr=1e-2)

# Generate dataset
class ODEData(Dataset):
    def __init__(self, func_type='additive', num_samples=100):
        self.xs = torch.full([num_samples], fill_value=1)
        self.us = torch.linspace(1, num_samples, num_samples)

        # Choose which function to use
        if func_type == 'additive':
            def f(x, t, u):
                return np.sin(x) + u
        else:
            def f(x, t, u):
                return u * np.sin(x)

        data = []
        for y0, u0 in zip(self.xs, self.us.numpy()):
            data.append(odeint(f, y0=y0, t=np.linspace(1, num_samples, num_samples), args=(u0,)))
        self.data = torch.from_numpy(np.stack(data)).float()

        self.us = self.us.reshape([num_samples, 1])

        self.data = self.data
        self.us = self.us

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.data[idx], self.us[idx]


dataset = ODEData(num_samples=100)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train loop
num_epochs = 10000
for epoch in range(num_epochs):
    metrics = []
    for b_idx, batch in enumerate(dataloader):
        gt_seq, u = batch

        net.ode.set_us(u)

        preds = net(gt_seq[:, 0])
        loss = mse(preds, gt_seq)

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"=> Epoch [{epoch}/{num_epochs}]: {loss.item()}")

    # plt.plot(gt_seq[0].detach().cpu().numpy())
    # plt.plot(preds[0].detach().cpu().numpy())
    # plt.legend(['GT', 'Pred'])
    # plt.show()


# Test loop
with torch.no_grad():
    net.ode.set_us(dataset.us)
    preds = net(dataset.data[:, 0])

    plt.figure()
    plt.plot(dataset.data[:, :, 0], linestyle='--', c='k')
    plt.plot(preds[:, :, 0], c='b')
    plt.show()