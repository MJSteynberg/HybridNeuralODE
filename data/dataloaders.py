#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ziqian Li
"""

import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io


class Dissipative(Dataset):
    def __init__(self, filename):
        mat_contents = scipy.io.loadmat(filename)
        self.t = torch.tensor(mat_contents['t'], dtype=torch.float32).squeeze()
        self.u = torch.tensor(mat_contents['u'], dtype=torch.float32).squeeze()

    def __getitem__(self, index):
        return self.t[index], self.u[index, :, :]

    def length_t(self):
        return len(self.t)

    def length_u(self):
        return self.u.shape[1]


class Anisotropic(Dataset):
    def __init__(self, filename):
        mat_contents = scipy.io.loadmat(filename)
        self.t = torch.tensor(mat_contents['t'], dtype=torch.float32).squeeze()
        self.x = torch.tensor(mat_contents['x'], dtype=torch.float32).squeeze()
        self.u = torch.tensor(mat_contents['u2'], dtype=torch.float32).squeeze()

    def __getitem__(self, index):
        return self.t[index], self.x, self.u[index, :]

    def __len__(self):
        return len(self.t)


class Nonlinear(Dataset):
    def __init__(self, filename):
        mat_contents = scipy.io.loadmat(filename)
        self.t = torch.tensor(mat_contents['t'], dtype=torch.float32).squeeze()
        self.x = torch.tensor(mat_contents['x'], dtype=torch.float32).squeeze()
        self.u = torch.tensor(mat_contents['unl1'], dtype=torch.float32).squeeze()

    def __getitem__(self, index):
        return self.t[index], self.x, self.u[index, :]

    def __len__(self):
        return len(self.t)


class Heat(Dataset):
    def __init__(self, filename):
        mat_contents = scipy.io.loadmat(filename)
        self.t = torch.tensor(mat_contents['t'], dtype=torch.float32).squeeze()
        self.x = torch.tensor(mat_contents['x'], dtype=torch.float32).squeeze()
        # Dictionary to store tensors
        self.variables = {}

        n = 20
        # Generate variable names
        variable_names = [f'u{i}' for i in range(1, n + 1)]

        # Load each variable as a tensor
        for var_name in variable_names:
            if var_name in mat_contents:
                # Convert to tensor and store in the dictionary
                self.variables[var_name] = torch.tensor(mat_contents[var_name], dtype=torch.float32).squeeze()
            else:
                print(f"Warning: {var_name} not found in the file.")

        # combine all solutions to [5, 10, 100]
        self.u = torch.stack([self.variables[f'u{i}'] for i in range(1, n + 1)], dim=0)
        self.u_test = torch.tensor(mat_contents['u_test'], dtype=torch.float32).squeeze()

        del self.variables

    def __getitem__(self, index):
        return self.t[index], self.x, self.u[:, index, :], self.u_test

    def __len__(self):
        return len(self.t)
