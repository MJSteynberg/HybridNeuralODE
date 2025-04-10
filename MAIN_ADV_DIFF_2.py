


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Code
"""

from data.dataloaders import DataLoader_Scalar
from models.generate_data import advectiondiffusion
import torch
from torch.utils.data import DataLoader, Subset
from models.training import Trainer
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

    # High definition solution
    data_high = np.load(folder + '/phys.npy')
    data_high = torch.from_numpy(data_high).float().to(device)

    # Obtain variables from data
    u_train = data.u[:, indices, :]
    grid, u0 = create_grid(-L//2, L//2, L/(nX-1))
    data_dim = data.u.shape[2]
    T = data.t[-1]
    num_steps = data.length_t()
    dt = data.t[1] - data.t[0]

    # create models, optimizers and schedulers
    node = NeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='relu', T=T, time_steps=num_steps).to(device)
    phys = AdvectionDiffusion(device, L, nX, dt, num_steps, num_gaussians_alpha, num_gaussians_kappa, alpha = alpha, kappa = kappa)

    optimizer_node = torch.optim.Adam(node.dynamics.parameters(), lr=learning_rate)
    optimizer_phys = torch.optim.Adam(phys.parameters(), lr=1e-3, betas = (0.5, 0.99999))

    scheduler_node = torch.optim.lr_scheduler.OneCycleLR(optimizer_node, max_lr=learning_rate, steps_per_epoch=1, epochs=num_epochs)
    scheduler_phys = torch.optim.lr_scheduler.OneCycleLR(optimizer_phys, max_lr=1e-1, steps_per_epoch=1, epochs=num_epochs)

    # create trainer 
    trainer = Trainer(node, phys, optimizer_node, optimizer_phys, scheduler_node, scheduler_phys, device, grid=grid, interaction=interaction, data_high=data_high, u0=u0)
    return u_train, indices, u0, trainer

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

    if input("Regenerate data? (y/n)") == "y":
        print("---------------------------------")
        print("--------Generating data----------")
        print("---------------------------------")
        advectiondiffusion(flag="full", L = 10)


    # Common parameters: 
    split = 0.6
    nX = 21
    L = 10
    folder = 'data/adv_diff/full'
    num_gaussians_alpha = 1
    num_gaussians_kappa = 1
    alpha_real = torch.tensor([3,  1, 1, 1.0]).float() 
    kappa_real = torch.tensor([2.5, -2, -2, 1.0]).float()

    alpha = torch.tensor([1, 1.4, -1.3, 1.0]).float().to(device) # [Amplitude, x0, y0, sigma]
    kappa = torch.tensor([2, -1.8, -1.1, 1.0]).float().to(device) # [Amplitude, x0, y0, sigma]
    hidden_dim = 1000
    learning_rate = 1e-3
    num_epochs = 3000

    u_train, indices, u0, trainer_hybrid = setup(split, nX, L, folder, num_gaussians_alpha, num_gaussians_kappa, alpha.clone(), kappa.clone(), hidden_dim, learning_rate, num_epochs, interaction = True)

    if input("Train model? (y/n)") == "y":
        _,_,_,trainer_phys = setup(split, nX, L, folder, num_gaussians_alpha, num_gaussians_kappa, alpha.clone(), kappa.clone(), hidden_dim, learning_rate, num_epochs, interaction = False)
        print("--------------------------------")
        print("--------Setup finished----------")
        print("--------------------------------")
        print("------- Physics training: ------")
        print("--------------------------------")
        params_phys, _, error_phys = trainer_phys.train(u_train, num_epochs)
        print("--------------------------------")
        print("------- Hybrid training: -------")
        print("--------------------------------")
        params_hybrid, error_hybrid_syn, error_hybrid_phys = trainer_hybrid.train(u_train, num_epochs)
        
        print("--------------------------------")
        print("-------Training finished--------")
        print("--------------------------------")




        # Save the actual models
        torch.save(trainer_hybrid.model_node.state_dict(), 'models/adv_diff/node.pt')
        torch.save(trainer_hybrid.model_phys.state_dict(), 'models/adv_diff/phys.pt')
    nX = 120
    data = DataLoader_Scalar(device, folder)
    train_size = int(split*data.length_u())
    indices = torch.randperm(data.length_u())[:train_size]
    device = torch.device('cpu')
    # Obtain variables from data
    L = 10
    u_train = data.u[:, indices, :]
    grid, u0 = create_grid(-L//2, L//2, L/(nX-1))
    data_dim = data.u.shape[2]
    T = data.t[-1]
    num_steps = data.length_t()
    dt = data.t[1] - data.t[0]
    # create models, optimizers and schedulers
    node = NeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='relu', T=T, time_steps=num_steps).to(device)
    #phys = AdvectionDiffusion(device, L, nX, dt, num_steps, num_gaussians_alpha, num_gaussians_kappa, alpha = alpha, kappa = kappa)
    node.load_state_dict(torch.load('models/adv_diff/node.pt'))
    #phys.load_state_dict(torch.load('models/adv_diff/phys.pt'))
    L = 14
    grid, u0 = create_grid(-L//2, L//2, L/(nX-1))
    #u_phys = phys(u0)
    # as input we need (x,y,u) using the grid
    x = grid[:, 0].reshape(-1,1)
    y = grid[:, 1].reshape(-1,1)
    u = u0.reshape(-1,1)
    print(x.shape, y.shape, u.shape, torch.cat([x, y, u], dim=1).shape)
    u_node = node.f(torch.cat([x, y, u], dim=1)).detach().numpy()

    
    # Scatter plot of the results
    fig, ax = plt.subplots(1, 5, figsize=(15, 5))
    ax[0].scatter(u_node[0,:,0], u_node[0,:,1], c=u_node[0,:,2], cmap='viridis')
    ax[1].scatter(u_node[20,:,0], u_node[20,:,1], c=u_node[20,:,2], cmap='viridis')
    ax[2].scatter(u_node[40,:,0], u_node[40,:,1], c=u_node[40,:,2], cmap='viridis')
    ax[3].scatter(u_node[60,:,0], u_node[60,:,1], c=u_node[60,:,2], cmap='viridis')
    ax[4].scatter(u_node[80,:,0], u_node[80,:,1], c=u_node[80,:,2], cmap='viridis')
    for i in range(5):
        ax[i].set_xlim(-3, 3)
        ax[i].set_ylim(-3, 3)

    plt.savefig('figures/adv_diff_node.png')
