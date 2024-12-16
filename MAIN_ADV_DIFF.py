#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Code
"""

from data.dataloaders import DataLoader_Scalar
import torch
from torch.utils.data import DataLoader, Subset
from HybridNeuralODE.models.training import Trainer
from models.neural_odes import NeuralODE
from models.plot import plot_results_separate
from models.FEM import AdvectionDiffusion
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

device = torch.device('cuda')

def setup(split, nX, L, folder, num_gaussians_alpha = 1, num_gaussians_kappa = 1, alpha = None, kappa = None, hidden_dim = 1000, learning_rate = 1e-3, num_epochs = 2000, interaction = True):
    # Load data and split
    data = DataLoader_Scalar(device, folder)
    train_size = int(split*data.length_u())
    indices = torch.randperm(data.length_u())[:train_size]

    # Obtain variables from data
    u_train = data.u[:, indices, :]
    grid, u0 = create_grid(-L//2, L//2, 6/(nX-1))
    data_dim = data.u.shape[2]
    T = data.t[-1]
    num_steps = data.length_t()
    dt = data.t[1] - data.t[0]

    # create models, optimizers and schedulers
    node = NeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='relu', T=T, time_steps=num_steps).to(device)
    phys = AdvectionDiffusion(device, L, nX, dt, num_steps, num_gaussians_alpha, num_gaussians_kappa, alpha = alpha, kappa = kappa)

    optimizer_node = torch.optim.Adam(node.dynamics.parameters(), lr=learning_rate)
    optimizer_phys = torch.optim.Adam(phys.parameters(), lr=1e-1)
    scheduler_node = torch.optim.lr_scheduler.OneCycleLR(optimizer_node, max_lr=learning_rate, steps_per_epoch=1, epochs=num_epochs)
    scheduler_phys = torch.optim.lr_scheduler.LambdaLR(optimizer_phys, lr_lambda=lambda epoch: 1.0e-1) #Just constant

    # create trainer 
    trainer = Trainer(node, phys, optimizer_node, optimizer_phys, scheduler_node, scheduler_phys, device, grid=grid, interaction=interaction)
    return u_train, u0, trainer

def create_grid(start, end, step):
    x = torch.arange(start, end + step, step, device=device)
    y = torch.arange(start, end + step, step, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    grid = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), torch.zeros((grid_x.numel(), 1), device=device)], dim=1)
    x, y = grid[:, 0], grid[:, 1]
    u0 = - torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)) + torch.exp(-((x - 1) ** 2 + (y - 1) ** 2))
    plate_length = int(np.sqrt(grid.shape[0]))
    u0_reshaped = u0.reshape(plate_length, plate_length)

    return grid, u0_reshaped



if __name__ == '__main__':

    # Common parameters: 
    split = 0.2
    nX = 21
    L = 6
    folder = 'data/adv_diff'
    num_gaussians_alpha = 1
    num_gaussians_kappa = 1
    alpha = torch.tensor([1, 1.4, -1.3, 1.0]).float().to(device) # [Amplitude, x0, y0, sigma]
    kappa = torch.tensor([1, -2, -1.1, 1.0]).float().to(device) # [Amplitude, x0, y0, sigma]
    hidden_dim = 1000
    learning_rate = 1e-3
    num_epochs = 10

    u_train, u0, trainer_hybrid = setup(split, nX, L, folder, num_gaussians_alpha, num_gaussians_kappa, alpha, kappa, hidden_dim, learning_rate, num_epochs, interaction = True)
    _,_,trainer_phys = setup(split, nX, L, folder, num_gaussians_alpha, num_gaussians_kappa, alpha, kappa, hidden_dim, learning_rate, num_epochs, interaction = False)
    print("--------------------------------")
    print("--------Setup finished----------")
    print("--------------------------------")
    params_hybrid = trainer_hybrid.train(u_train, u0, num_epochs)
    params_phys = trainer_phys.train(u_train, u0, num_epochs)
    print("--------------------------------")
    print("-------Training finished--------")
    print("--------------------------------")

    # Save all parameters to parameters folder using date and time in name
    df = pd.DataFrame(np.stack([params_hybrid.flatten(), params_phys.flatten()]).T, columns=["params_hybrid", "params_phys"])
    
    # Get the current date and time to include in the filename
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Save the DataFrame to CSV with the timestamp in the filename
    df.to_csv(f'parameters/adv_diff/param_{timestamp}.csv', index=False)

