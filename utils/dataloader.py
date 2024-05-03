import matplotlib.pyplot as plt
import torch
import numpy as np

from scipy.integrate import odeint
from torch.utils.data import Dataset


class MetaODEData(Dataset):
    def __init__(self, args, shuffle=True):
        self.func_type = args.func_type

        self.shuffle = shuffle
        self.k_shot = args.k_shot
        self.ts = np.linspace(-1, -0.5, args.timesteps)
        self.xs = torch.linspace(1, 5, args.num_tasks)

        # Choose which function to use
        if args.func_type == 'additive':
            self.us = torch.linspace(-4, 4, args.num_samples)

            def f(x, t, u):
                return np.sin(x) + u
        else:
            self.us = torch.linspace(-2, 2, args.num_samples)

            def f(x, t, u):
                return u * np.sin(x)

        data = []
        labels = []
        us = []
        for label, x0 in enumerate(self.xs.numpy()):
            for u0 in zip(self.us.numpy()):
                data.append(odeint(f, y0=x0, t=self.ts, args=(u0,)))
                us.append(u0)
                labels.append(label)
        self.data = torch.from_numpy(np.stack(data)).float()
        self.labels = torch.from_numpy(np.stack(labels)).float()
        self.us = torch.from_numpy(np.stack(us)).float()

        print(f"=> Data {self.data.shape}")
        print(f"=> Labels {self.labels.shape}")
        print(f"=> Controls {self.us.shape}")

        # Get labels of environments
        self.label_idx = {}
        for label in np.unique(self.labels):
            idx = np.where(self.labels == label)[0]
            self.label_idx[label] = idx

        # Get data dimensions
        self.sequences, self.timesteps, self.dim = self.data.shape
        self.split()

    def __len__(self):
        return self.qry_idx.shape[0]

    def __getitem__(self, idx):
        """ Get signal and support set """
        label_qry = int(self.labels[self.qry_idx[idx]])
        control_qry = self.us[self.qry_idx[idx]]
        signal_qry = self.data[self.qry_idx[idx], :]
        signal_spt = self.data[self.spt_idx[label_qry], :]

        return torch.Tensor([idx]), signal_qry, signal_spt, label_qry, control_qry

    def split(self):
        self.spt_idx = {}
        self.qry_idx = []
        for label_id, samples in self.label_idx.items():
            sample_idx = np.arange(0, len(samples))
            if len(samples) < self.k_shot:
                self.spt_idx[label_id] = samples
            else:
                if self.shuffle:
                    np.random.shuffle(sample_idx)
                    spt_idx = np.sort(sample_idx[0:self.k_shot])
                else:
                    spt_idx = sample_idx[0:self.k_shot]
                self.spt_idx[label_id] = samples[spt_idx]

            # Build mask of support indices to remove from current query indices
            mask = np.full(len(samples), True, dtype=bool)
            mask[spt_idx] = False

            # Append remaining samples to query indices
            self.qry_idx.extend(samples[mask])

        self.qry_idx = np.array(self.qry_idx)
        self.qry_idx = np.sort(self.qry_idx)


class SingleODEData(Dataset):
    def __init__(self, args, x_initials, u_start, u_end):
        self.func_type = args.func_type

        self.ts = np.linspace(-1, 0, args.timesteps)
        self.xs = torch.Tensor(x_initials).float()

        # Choose which function to use
        if args.func_type == 'additive':
            self.us = torch.linspace(-4, 4, args.num_samples)

            def f(x, t, u):
                return np.sin(x) + u
        else:
            self.us = torch.linspace(-2, 2, args.num_samples)

            def f(x, t, u):
                return u * np.sin(x)

        data = [[] for _ in range(len(x_initials))]
        us = [[] for _ in range(len(x_initials))]
        for x_idx, x0 in enumerate(self.xs.numpy()):
            for u0 in zip(self.us.numpy()):
                data[x_idx].append(odeint(f, y0=x0, t=self.ts, args=(u0,)))
                us[x_idx].append(u0)

        data = [torch.from_numpy(np.stack(d)) for d in data]
        us = [torch.from_numpy(np.stack(u)) for u in us]

        self.data = torch.concatenate(data, dim=-1).float()
        self.us = torch.concatenate(us, dim=-1).float()
        print(f"=> Data {self.data.shape}")
        print(f"=> Controls {self.us.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx, self.data[idx], self.us[idx]


""" Generalization Experiments """


class MetaPercentODEData(Dataset):
    def __init__(self, args, u_start, u_end, bounds=(0, 100), shuffle=True, is_interpolation=False, is_train=True):
        self.func_type = args.func_type

        self.shuffle = shuffle
        self.k_shot = args.k_shot
        self.ts = np.linspace(-1, 0, args.timesteps)
        self.xs = torch.linspace(1, 5, args.num_tasks)
        self.us = torch.linspace(u_start, u_end, args.num_samples)

        if is_interpolation is True:
            self.us = self.us + (0.2 * torch.rand_like(self.us))

        def f(x, t, u):
            return u * np.sin(x)

        data, labels, us = [], [], []
        for label, x0 in enumerate(self.xs.numpy()):
            for u0 in zip(self.us.numpy()):
                data.append(odeint(f, y0=x0, t=self.ts, args=(u0,)))
                us.append(u0)
                labels.append(label)
        self.data = torch.from_numpy(np.stack(data)).float()
        self.labels = torch.from_numpy(np.stack(labels)).float()
        self.us = torch.from_numpy(np.stack(us)).float()

        self.data = self.data.reshape([args.num_tasks, args.num_samples, args.timesteps, 1])
        self.labels = self.labels.reshape([args.num_tasks, args.num_samples, 1])
        self.us = self.us.reshape([args.num_tasks, args.num_samples, 1])

        if is_train is True or is_interpolation is True:
            self.data = self.data[:, bounds[0]:bounds[1]]
            self.labels = self.labels[:, bounds[0]:bounds[1]]
            self.us = self.us[:, bounds[0]:bounds[1]]
        # If it is testing, get the extrapolation bounds
        elif is_train is False and is_interpolation is False:
            self.data = torch.concatenate((self.data[:, bounds[0] - args.extrapolation_bound_window:bounds[0]], self.data[:, bounds[1]:bounds[1] + args.extrapolation_bound_window]))
            self.labels = torch.concatenate((self.labels[:, bounds[0] - args.extrapolation_bound_window:bounds[0]], self.labels[:, bounds[1]:bounds[1] + args.extrapolation_bound_window]))
            self.us = torch.concatenate((self.us[:, bounds[0] - args.extrapolation_bound_window:bounds[0]], self.us[:, bounds[1]:bounds[1] + args.extrapolation_bound_window]))

        self.data = self.data.reshape([-1, args.timesteps, 1])
        self.labels = self.labels.reshape([-1, 1])
        self.us = self.us.reshape([-1, 1])

        print(f"=> Data {self.data.shape}")
        print(f"=> Labels {self.labels.shape}")
        print(f"=> Controls {self.us.shape}")

        # Get labels of environments
        self.label_idx = {}
        for label in np.unique(self.labels):
            idx = np.where(self.labels == label)[0]
            self.label_idx[label] = idx

        # Get data dimensions
        self.sequences, self.timesteps, self.dim = self.data.shape
        self.split()

    def __len__(self):
        return self.qry_idx.shape[0]

    def __getitem__(self, idx):
        """ Get signal and support set """
        label_qry = int(self.labels[self.qry_idx[idx]])
        control_qry = self.us[self.qry_idx[idx]]
        signal_qry = self.data[self.qry_idx[idx], :]
        signal_spt = self.data[self.spt_idx[label_qry], :]

        return torch.Tensor([idx]), signal_qry, signal_spt, label_qry, control_qry

    def split(self):
        self.spt_idx = {}
        self.qry_idx = []
        for label_id, samples in self.label_idx.items():
            sample_idx = np.arange(0, len(samples))
            if len(samples) < self.k_shot:
                self.spt_idx[label_id] = samples
            else:
                if self.shuffle:
                    np.random.shuffle(sample_idx)
                    spt_idx = np.sort(sample_idx[0:self.k_shot])
                else:
                    spt_idx = sample_idx[0:self.k_shot]
                self.spt_idx[label_id] = samples[spt_idx]

            # Build mask of support indices to remove from current query indices
            mask = np.full(len(samples), True, dtype=bool)
            mask[spt_idx] = False

            # Append remaining samples to query indices
            self.qry_idx.extend(samples[mask])

        self.qry_idx = np.array(self.qry_idx)
        self.qry_idx = np.sort(self.qry_idx)


class PercentODEData(Dataset):
    def __init__(self, args, x_initial, u_start, u_end, bounds=(0, 100), is_interpolation=False, is_train=True):
        self.func_type = args.func_type

        self.ts = np.linspace(-1, 0, args.timesteps)
        self.xs = torch.Tensor(x_initial).float()
        self.us = torch.linspace(u_start, u_end, args.num_samples)

        if is_interpolation is True:
            self.us = self.us + (0.2 * torch.rand_like(self.us))

        # Choose which function to use
        if args.func_type == 'additive':
            def f(x, t, u):
                return np.sin(x) + u
        else:
            def f(x, t, u):
                return u * np.sin(x)

        avgs, data, us = [], [], []
        for u0 in zip(self.us.numpy()):
            sol = odeint(f, y0=x_initial, t=self.ts, args=(u0,))
            data.append(sol)
            us.append(u0)
            avgs.append(np.mean(sol))

        data = [torch.from_numpy(np.stack(d)) for d in data]
        us = [torch.from_numpy(np.stack(u)) for u in us]

        plt.figure()
        plt.plot(range(args.num_samples), avgs, c='k')
        plt.plot(range(bounds[0] - args.extrapolation_bound_window, bounds[1] + args.extrapolation_bound_window), [avgs[i] for i in range(bounds[0] - args.extrapolation_bound_window, bounds[1] + args.extrapolation_bound_window)], c='b')
        plt.scatter(range(bounds[0], bounds[1]), [avgs[i] for i in range(bounds[0], bounds[1])], marker='x')

        # TODO REMOVE
        plt.plot(range(60, 100), [avgs[i] for i in range(60, 100)], c='r')
        plt.scatter(range(70, 90), [avgs[i] for i in range(70, 90)], marker='x')


        plt.title("A4) Avg. Signal Value over the Control Dimension", fontdict={"fontweight": 25})
        plt.xlabel("Value of the Control Variable C")
        plt.ylabel("Average Signal Value")
        plt.xticks(
            [0, 24, 49, 74, 99],
            np.round(np.array([self.us[0], self.us[24], self.us[49], self.us[74], self.us[99]]), 2)
        )
        plt.show()
        plt.close()

        self.data = torch.stack(data).float()
        self.us = torch.stack(us).float()

        plt.figure()
        plt.plot(range(20), self.data[bounds[0] - args.extrapolation_bound_window:bounds[0], :, 0].T, c='b')
        plt.plot(range(20), self.data[bounds[0]:bounds[1], :, 0].T, c='gray', alpha=0.5)
        plt.plot(range(20), self.data[bounds[1]:bounds[1] + args.extrapolation_bound_window, :, 0].T, c='b')

        plt.plot(range(20), self.data[60:70, :, 0].T, c='r')
        plt.plot(range(20), self.data[70:90, :, 0].T, c='gray', alpha=0.5)
        plt.plot(range(20), self.data[90:100, :, 0].T, c='r')

        plt.xticks(range(20))
        plt.title("A3) sin(x) * u", fontdict={"fontweight": 25})
        plt.ylabel("Function Value")
        plt.xlabel("Timestep")
        plt.show()
        plt.close()

        if is_train is True or is_interpolation is True:
            self.data = self.data[bounds[0]:bounds[1]]
            self.us = self.us[bounds[0]:bounds[1]]
        # If it is testing, get the extrapolation bounds
        elif is_train is False and is_interpolation is False:
            self.data = torch.concatenate((self.data[bounds[0] - args.extrapolation_bound_window:bounds[0]], self.data[bounds[1]:bounds[1] + args.extrapolation_bound_window]))
            self.us = torch.concatenate((self.us[bounds[0] - args.extrapolation_bound_window:bounds[0]], self.us[bounds[1]:bounds[1] + args.extrapolation_bound_window]))

        print(f"=> Data {self.data.shape}")
        print(f"=> Controls {self.us.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx, self.data[idx], self.us[idx]
