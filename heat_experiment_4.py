#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Code
"""

from data.dataloaders import Dissipative
import torch
from torch.utils.data import DataLoader, Subset
from models.training import Trainer
import models.training_no_interaction as tni
from models.neural_odes import NeuralODE
from models.plot import plot_results_separate
from models.FEM import heatsolve
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')


def train_hybrid():
    # =============================================================================
    # DATA
    # =============================================================================
    filename_data = 'Experiments/4/u4new.mat'
    datasets = Dissipative(filename_data)
    t, u = datasets[:]
    

    # Create indices and split for train and test data
    train_size = int(0.6 * datasets.length_u())
    indices = torch.randperm(datasets.length_u())
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    u_train = u[:, train_indices, :].to(device)
    u_test = u[:, test_indices, :].to(device)

    # Define a grid of points within [-3, 3] with an added column of zeros
    nX = 21
    grid, u0 = create_grid(-3, 3, 6/(nX-1))

    # =============================================================================
    # SETUP
    # =============================================================================
    T, num_steps = max(t).item(), datasets.length_t()
    dt = T / (num_steps - 1)
    data_dim = u.shape[2]

    # Choose optimizer settings
    L1 = False
    weight_decay = 0.0 if L1 else 0.01 * dt

    # =============================================================================
    # MODEL SETUP
    # =============================================================================
    anode = NeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='relu',
                      architecture='bottleneck', T=T, time_steps=num_steps).to(device)

    heat = heatsolve(device, T=T, nX=nX, time_steps=num_steps, param_x=1, param_y=2).to(device)

    # Optimizers
    optimizer_anode = torch.optim.Adam(anode.dynamics.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_heat = torch.optim.Adam(heat.parameters(), lr=1e-1) 
    scheduler_anode = torch.optim.lr_scheduler.OneCycleLR(optimizer_anode, max_lr=learning_rate, steps_per_epoch=1, epochs=num_epochs)
    scheduler_heat = torch.optim.lr_scheduler.LambdaLR(optimizer_heat, lr_lambda=lambda epoch: 1.0e-1) #Just constant
    # =============================================================================
    # TRAINING
    # =============================================================================
    trainer_anode = Trainer(anode, heat, optimizer_anode, optimizer_heat, scheduler_anode, scheduler_heat, device, reg=reg, grid=grid)
    alpha_list = trainer_anode.train(u_train, u0, num_epochs)

    # =============================================================================
    # TESTING
    # =============================================================================
    def test_prediction(data, initial_condition):
        pred, traj_pred = anode(initial_condition)
        pred_save = torch.cat((initial_condition.unsqueeze(0), traj_pred), dim=0)
        return pred_save.detach().cpu().numpy()

    # u_test_NN = test_prediction(u_test[0, :, :], u_test[0, :, :])
    # u_train_NN = test_prediction(u_train[0, :, :], u_train[0, :, :])

    # # Convert and combine real values for plotting
    # u_real = np.concatenate((u_train.detach().cpu().numpy(), u_test.detach().cpu().numpy()), axis=1)

    # plot_results_separate(u_real, u_train_NN, u_test_NN, dim=3, plot_type='Simulation')
    return alpha_list

def train():
    # =============================================================================
    # DATA
    # =============================================================================
    filename_data = 'Experiments/4/u4new.mat'
    datasets = Dissipative(filename_data)
    t, u = datasets[:]

    # Create indices and split for train and test data
    train_size = int(0.6 * datasets.length_u())
    indices = torch.randperm(datasets.length_u())
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    u_train = u[:, train_indices, :].to(device)
    u_test = u[:, test_indices, :].to(device)

    # Define a grid of points within [-3, 3] with an added column of zeros
    grid, u0 = create_grid(-3, 3, 0.3)

    # =============================================================================
    # SETUP
    # =============================================================================
    T, num_steps = max(t).item(), datasets.length_t()
    dt = T / (num_steps - 1)
    data_dim = u.shape[2]

    # Choose optimizer settings
    L1 = False
    weight_decay = 0.0 if L1 else 0.01 * dt

    # =============================================================================
    # MODEL SETUP
    # =============================================================================
    anode = NeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='relu',
                      architecture='bottleneck', T=T, time_steps=num_steps).to(device)

    heat = heatsolve(device, T=T, nX=11, time_steps=num_steps, param_x=2, param_y=1).to(device)

    # Optimizers
    optimizer_anode = torch.optim.Adam(anode.dynamics.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_heat = torch.optim.Adam(heat.parameters(), lr=1e-1)
    scheduler_anode = torch.optim.lr_scheduler.OneCycleLR(optimizer_anode, max_lr=learning_rate, steps_per_epoch=1, epochs=num_epochs)
    scheduler_heat = torch.optim.lr_scheduler.LambdaLR(optimizer_heat, lr_lambda=lambda epoch: 1.0e-1) #Just constant LR
    # =============================================================================
    # TRAINING
    # =============================================================================
    trainer_anode = Trainer(anode, heat, optimizer_anode, optimizer_heat, scheduler_anode, scheduler_heat, device, reg=reg, grid=grid)
    alpha_list = trainer_anode.train(u_train, u0, num_epochs)

    # =============================================================================
    # TESTING
    # =============================================================================
    # def test_prediction(data, initial_condition):
    #     pred, traj_pred = anode(initial_condition)
    #     pred_save = torch.cat((initial_condition.unsqueeze(0), traj_pred), dim=0)
    #     return pred_save.detach().cpu().numpy()

    # u_test_NN = test_prediction(u_test[0, :, :], u_test[0, :, :])
    # u_train_NN = test_prediction(u_train[0, :, :], u_train[0, :, :])

    # # Convert and combine real values for plotting
    # u_real = np.concatenate((u_train.detach().cpu().numpy(), u_test.detach().cpu().numpy()), axis=1)

    # plot_results_separate(u_real, u_train_NN, u_test_NN, dim=3, plot_type='Simulation')
    return alpha_list


def create_grid(start, end, step):
    x = torch.arange(start, end + step, step, device=device)
    y = torch.arange(start, end + step, step, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    grid = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), torch.zeros((grid_x.numel(), 1), device=device)],
                     dim=1)
    x_grid, y_grid = grid[:, 0], grid[:, 1]
    u0 = 5 * torch.exp(-5 * ((x_grid + 1) ** 2 + (y_grid + 1) ** 2)) + 5 * torch.exp(
        -5 * ((x_grid - 1) ** 2 + (y_grid - 1) ** 2))
    plate_length = int(np.sqrt(grid.shape[0]))
    u0_reshaped = u0.reshape(plate_length, plate_length)

    return grid, u0_reshaped



if __name__ == '__main__':
    num_epochs = 2000
    hidden_dim = 1000
    learning_rate = 1e-3
    reg = 'l1'
    
    for i in range(1):
        alpha_h, loss_h = train_hybrid()
        np.save(f'Experiments/4/alpha_h_{i}.npy', alpha_h.detach())
        np.save(f'Experiments/4/loss_h_{i}.npy', loss_h)
        alpha_, loss_ = train()
        np.save(f'Experiments/4/alpha_{i}.npy', alpha_.detach())
        np.save(f'Experiments/4/loss_{i}.npy', loss_)
        
    for i in range(1):
        alpha_hybrid = np.load(f'Experiments/4/alpha_h_{i}.npy')
        alpha = np.load(f'Experiments/4/alpha_{i}.npy')
        fig = plt.figure()
        filename_data = 'Experiments/4/u.mat'
        datasets = Dissipative(filename_data)
        t, u = datasets[:]
        
        grid, u0 = create_grid(-3, 3, 0.6)

        
        grid = grid.reshape(11, 11, 3)
        
        
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-3, 3)
        ax1.imshow(alpha[-1].T, extent=(-3, 3, -3, 3), origin='lower', interpolation='bilinear', cmap='coolwarm')
        ax2.imshow(alpha_hybrid[-1].T, extent=(-3, 3, -3, 3), origin='lower', interpolation='bilinear', cmap='coolwarm')
        # plot the value at the center of each of the four corners
        ax1.text(-2, -2, round(alpha[-1][1, 1],2), ha='center', va='center', color='black')
        ax1.text(-2, 2, round(alpha[-1][1, -2],2), ha='center', va='center', color='black')
        ax1.text(2, -2, round(alpha[-1][-2, 1],2), ha='center', va='center', color='black')
        ax1.text(2, 2, round(alpha[-1][-2, -2],2), ha='center', va='center', color='black')
        
        ax2.text(-2, -2, round(alpha_hybrid[-1][1, 1],2), ha='center', va='center', color='black')
        ax2.text(-2, 2, round(alpha_hybrid[-1][1, -2],2), ha='center', va='center', color='black')
        ax2.text(2, -2, round(alpha_hybrid[-1][-2, 1],2), ha='center', va='center', color='black')
        ax2.text(2, 2, round(alpha_hybrid[-1][-2, -2],2), ha='center', va='center', color='black')
        
        
        
        
        # plot the (xy) trajectories of the heat source
        print(u.shape)
        for j in range(0, 25):
            ax1.plot(u[:, j, 0].cpu().numpy(), u[:, j, 1].cpu().numpy(), c='k')
            ax2.plot(u[:, j, 0].cpu().numpy(), u[:, j, 1].cpu().numpy(), c='k')

        
        plt.savefig(f'Experiments/4/alpha_{i}.png', dpi = 300)
        
            

        

        
        
        