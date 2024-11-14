#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint

# Constants for maximum steps and activation functions
MAX_NUM_STEPS = 1000
activations = {'tanh': nn.Tanh(),
               'relu': nn.ReLU(),
               'sigmoid': nn.Sigmoid(),
               'leakyrelu': nn.LeakyReLU()}
architectures = {'inside': -1, 'outside': 0, 'bottleneck': 1}


class Dynamics(nn.Module):
    """Represents the dynamics f(theta(t), x(t)) in the neural ODE."""

    def __init__(self, device, data_dim, hidden_dim, augment_dim=0,
                 non_linearity='tanh', architecture='inside', T=10, time_steps=10):
        super(Dynamics, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim

        self.non_linearity = activations.get(non_linearity, nn.Tanh())
        self.architecture = architectures.get(architecture, -1)
        self.T = T
        self.time_steps = time_steps

        # Define layers with optimized initialization
        self.fc1_time_1 = nn.Linear(self.input_dim - 1, hidden_dim, bias=False)
        self.fc1_time_2 = nn.Linear(self.input_dim - 1, hidden_dim, bias=False)
        self.fc1_time_3 = nn.Linear(self.input_dim, hidden_dim, bias=False)

        self.b_time_1 = nn.Linear(1, hidden_dim)
        self.b_time_2 = nn.Linear(1, hidden_dim)
        self.b_time_3 = nn.Linear(1, hidden_dim)

        self.fc3_time_1 = nn.Linear(hidden_dim, 1, bias=False)
        self.fc3_time_2 = nn.Linear(hidden_dim, 1, bias=False)
        self.fc3_time_3 = nn.Linear(hidden_dim, 1, bias=False)

        # Efficient initialization
        self.apply(self._initialize_weights)

    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    def forward(self, t, x):
        """Computes f(x(t), u(t))."""
        # Pre-calculate linear transformations outside of main computational steps
        t = t.view(-1, 1)
        out_1 = self.non_linearity(self.fc1_time_1(x[:, :2]) + self.b_time_1(t))
        out_2 = self.non_linearity(self.fc1_time_2(x[:, :2]) + self.b_time_2(t))
        out_3 = self.non_linearity(self.fc1_time_3(x) + self.b_time_3(t))
        
        # Output concatenation and final transformation
        out = torch.cat((self.fc3_time_1(out_1), self.fc3_time_2(out_2), self.fc3_time_3(out_3)), dim=1)
        return out


class Semiflow(nn.Module):
    """Generates semiflow by solving x'(t) = f(theta(t), x(t))."""

    def __init__(self, device, dynamics, tol=1e-3, T=10, time_steps=10):
        super(Semiflow, self).__init__()
        self.device = device
        self.dynamics = dynamics
        self.tol = tol
        self.T = T
        self.time_steps = time_steps

    def forward(self, u, eval_times=None):
        dt = self.T / (self.time_steps - 1)
        integration_time = torch.tensor([0, self.T],
                                        dtype=u.dtype,
                                        device=self.device) if eval_times is None else eval_times

        # Efficiently solve ODE using pre-defined method
        out = odeint(self.dynamics,
                     u,
                     integration_time,
                     method='dopri5', rtol=1e-3, atol=1e-3)

        return out[1] if eval_times is None else out[1:]

    def trajectory(self, u, timesteps):
        integration_time = torch.linspace(0., self.T, timesteps, device=self.device)
        return self.forward(u, eval_times=integration_time)


class NeuralODE(nn.Module):
    """Returns the flowmap of the neural ODE."""

    def __init__(self, device, data_dim, hidden_dim, output_dim=2, augment_dim=0,
                 non_linearity='tanh', tol=1e-3, architecture='inside', T=10, time_steps=10):
        super(NeuralODE, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.tol = tol
        self.T = T
        self.time_steps = time_steps
        self.architecture = architecture

        self.dynamics = Dynamics(device, data_dim, hidden_dim, augment_dim, non_linearity, architecture, T, time_steps)
        self.flow = Semiflow(device, self.dynamics, tol, T, time_steps)
        self.linear_layer = nn.Linear(self.flow.dynamics.input_dim, output_dim)
        self.non_linearity = nn.Hardtanh(min_val=-1.0, max_val=1.0)

    def forward(self, u):
        features = self.flow(u)
        self.traj = self.flow.trajectory(u, self.time_steps)
        return features, self.traj
