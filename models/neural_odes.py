#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
@author2: MJ Steynberg
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


class Dynamics(nn.Module):
    """Represents the dynamics f(theta(t), x(t)) in the neural ODE."""

    def __init__(self, device, data_dim, hidden_dim, augment_dim=0,
                 non_linearity='tanh', T=10, time_steps=10):
        super(Dynamics, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim

        self.non_linearity = activations.get(non_linearity, nn.Tanh())
        self.T = T
        self.time_steps = time_steps

        # Define layers with optimized initialization
        
        self.fc1_time_1 = nn.ModuleList([nn.Linear(self.input_dim - 1, hidden_dim) for _ in range(time_steps)])
        self.fc1_time_2 = nn.ModuleList([nn.Linear(self.input_dim - 1, hidden_dim) for _ in range(time_steps)])
        self.fc1_time_3 = nn.ModuleList([nn.Linear(self.input_dim, hidden_dim) for _ in range(time_steps)])

        self.fc3_time_1 = nn.ModuleList([nn.Linear(hidden_dim, 1, bias=False) for _ in range(time_steps)])
        self.fc3_time_2 = nn.ModuleList([nn.Linear(hidden_dim, 1, bias=False) for _ in range(time_steps)])
        self.fc3_time_3 = nn.ModuleList([nn.Linear(hidden_dim, 1, bias=False) for _ in range(time_steps)])
        
        

        # Efficient initialization
        self.apply(self._initialize_weights)

    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    def g(self, t, x): 
        """Computes g(x(t))."""
        t = int(t * (self.time_steps)/self.T)
        out_1 = self.non_linearity(self.fc1_time_1[t](x))
        out_2 = self.non_linearity(self.fc1_time_2[t](x))
        # Output concatenation and final transformation
        out = torch.cat((self.fc3_time_1[t](out_1), self.fc3_time_2[t](out_2)), dim=1)
        return out
        
        
    def f(self, t, x):
        """Computes f(x(t), u(t))."""
        
        out_1 = self.g(t, x[:,:2])
        t = int(t * (self.time_steps)/self.T)
        out_3 = self.non_linearity(self.fc1_time_3[t](x))
        # Output concatenation and final transformation
        out = torch.cat((out_1, self.fc3_time_3[t](out_3)), dim=1)
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

    def g(self, u, eval_times=None):
        integration_time = torch.tensor([0, self.T], dtype=u.dtype, device=self.device) if eval_times is None else eval_times

        # Efficiently solve ODE using pre-defined method. Here Euler is required due to explicit layers
        out = odeint(self.dynamics.g, u[:, :2], integration_time, method='euler')

        return out[1] if eval_times is None else out[1:]
    
    def f(self, u, eval_times=None):
        integration_time = torch.tensor([0, self.T], dtype=u.dtype, device=self.device) if eval_times is None else eval_times

        # Efficiently solve ODE using pre-defined method
        out = odeint(self.dynamics.f, u, integration_time, method='euler')

        return out[1] if eval_times is None else out[1:]
    
    def sensor_trajectory(self, u, timesteps):
        integration_time = torch.linspace(0., self.T, timesteps, device=self.device)
        return self.g(u, eval_times=integration_time)
    
    def trajectory(self, u, timesteps):
        integration_time = torch.linspace(0., self.T, timesteps, device=self.device)
        return self.f(u, eval_times=integration_time)
    


class NeuralODE(nn.Module):
    """Returns the flowmap of the neural ODE."""

    def __init__(self, device, data_dim, hidden_dim, output_dim=2, augment_dim=0,
                 non_linearity='tanh', tol=1e-3, T=10, time_steps=10):
        super(NeuralODE, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.tol = tol
        self.T = T
        self.time_steps = time_steps

        self.dynamics = Dynamics(device, data_dim, hidden_dim, augment_dim, non_linearity, T, time_steps)
        self.flow = Semiflow(device, self.dynamics, tol, T, time_steps)
        self.linear_layer = nn.Linear(self.flow.dynamics.input_dim, output_dim)
        self.non_linearity = nn.Hardtanh(min_val=-1.0, max_val=1.0)

    def g(self, u):
        self.traj = self.flow.sensor_trajectory(u, self.time_steps)
        return self.traj
    
    def f(self, u):
        self.traj = self.flow.trajectory(u, self.time_steps)
        return self.traj
