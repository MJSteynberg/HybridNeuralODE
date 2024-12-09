#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Code
"""

from data.dataloaders import Dissipative
import torch
from torch.utils.data import DataLoader, Subset
from models.training_new import Trainer
from models.neural_odes_new import NeuralODE
from models.plot import plot_results_separate
from models.FEM import HeatEquation
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')


def train_hybrid(i):
    # =============================================================================
    # DATA
    # =============================================================================
    filename_data = 'Experiments/2materials/u.mat'
    u = np.load('Experiments/2materials/data.npy')
    u = torch.tensor(u, dtype=torch.float32)
    t = torch.linspace(0, 0.1, 100)
    
    length_u = u.shape[1]
    
    torch.set_rng_state(torch.manual_seed(42).get_state())
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(0, u.shape[1]):
    #     ax.plot(u[:, i, 0].detach().cpu().numpy(), u[:, i, 1].detach().cpu().numpy(), u[:, i, 2].detach().cpu().numpy())
    # plt.show()

    # Create indices and split for train and test data
    train_size = int(0.1*i * length_u)
    print(train_size)
    indices = torch.randperm(length_u)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    u_train = u[:, train_indices, :].to(device)
    u_test = u[:, test_indices, :].to(device)
    
    nX = 60
    # Define a grid of points within [-3, 3] with an added column of zeros
    grid, u0 = create_grid(-3, 3, 6/(nX-1))

    # =============================================================================
    # SETUP
    # =============================================================================
    T, num_steps = 0.2, 100
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
    param_x = 5
    param_y = 5
    heat = HeatEquation(device, 6, nX, dt, num_steps, param_x, param_y)
    # Optimizers
    optimizer_anode = torch.optim.Adam(anode.dynamics.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_heat = torch.optim.Adam(heat.parameters(), lr=1e-1)
    scheduler_anode = torch.optim.lr_scheduler.OneCycleLR(optimizer_anode, max_lr=learning_rate, steps_per_epoch=1, epochs=num_epochs)
    scheduler_heat = torch.optim.lr_scheduler.LambdaLR(optimizer_heat, lr_lambda=lambda epoch: 1.0e-1) #Just constant

    # =============================================================================
    # TRAINING
    # =============================================================================
    trainer_anode = Trainer(anode, heat, optimizer_anode, optimizer_heat, scheduler_anode, scheduler_heat, device, reg=reg, grid=grid, interaction=True)
    trainer_anode.train(u_train, u0, num_epochs)

    # # =============================================================================
    # # TESTING
    # # =============================================================================
    # def test_prediction(data, initial_condition):
    #     pred, traj_pred = anode(initial_condition)
    #     pred_save = torch.cat((initial_condition.unsqueeze(0), traj_pred), dim=0)
    #     return pred_save.detach().cpu().numpy()

    # u_test_NN = test_prediction(u_test[0, :, :], u_test[0, :, :])
    # u_train_NN = test_prediction(u_train[0, :, :], u_train[0, :, :])

    # # Convert and combine real values for plotting
    # u_real = np.concatenate((u_train.detach().cpu().numpy(), u_test.detach().cpu().numpy()), axis=1)

    # plot_results_separate(u_real, u_train_NN, u_test_NN, dim=3, plot_type='Simulation')
    return heat.parameters()

def train(i):
    # =============================================================================
    # DATA
    # =============================================================================
    filename_data = 'Experiments/2materials/u.mat'
    u = np.load('Experiments/2materials/data.npy')
    u = torch.tensor(u, dtype=torch.float32)
    t = torch.linspace(0, 0.1, 100)
    
    length_u = u.shape[1]
    
    torch.set_rng_state(torch.manual_seed(42).get_state())
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(0, u.shape[1]):
    #     ax.plot(u[:, i, 0].detach().cpu().numpy(), u[:, i, 1].detach().cpu().numpy(), u[:, i, 2].detach().cpu().numpy())
    # plt.show()

    # Create indices and split for train and test data
    train_size = int(0.1*i * length_u)
    print(train_size)
    indices = torch.randperm(length_u)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    u_train = u[:, train_indices, :].to(device)
    u_test = u[:, test_indices, :].to(device)
    
    nX = 60
    # Define a grid of points within [-3, 3] with an added column of zeros
    grid, u0 = create_grid(-3, 3, 6/(nX-1))

    # =============================================================================
    # SETUP
    # =============================================================================
    T, num_steps = 0.2, 100
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

    param_x = 5
    param_y = 5
    heat = HeatEquation(device, 6, nX, dt, num_steps, param_x, param_y)
    # Optimizers
    optimizer_anode = torch.optim.Adam(anode.dynamics.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_heat = torch.optim.Adam(heat.parameters(), lr=1e-1)
    scheduler_anode = torch.optim.lr_scheduler.OneCycleLR(optimizer_anode, max_lr=learning_rate, steps_per_epoch=1, epochs=num_epochs)
    scheduler_heat = torch.optim.lr_scheduler.LambdaLR(optimizer_heat, lr_lambda=lambda epoch: 1.0e-1) #Just constant
    # =============================================================================
    # TRAINING
    # =============================================================================
    trainer_anode = Trainer(anode, heat, optimizer_anode, optimizer_heat, scheduler_anode, scheduler_heat, device, reg=reg, grid=grid, interaction=False)
    trainer_anode.train(u_train, u0, num_epochs)

    # # =============================================================================
    # # TESTING
    # # =============================================================================
    # def test_prediction(data, initial_condition):
    #     pred, traj_pred = anode(initial_condition)
    #     pred_save = torch.cat((initial_condition.unsqueeze(0), traj_pred), dim=0)
    #     return pred_save.detach().cpu().numpy()

    # u_test_NN = test_prediction(u_test[0, :, :], u_test[0, :, :])
    # u_train_NN = test_prediction(u_train[0, :, :], u_train[0, :, :])

    # # Convert and combine real values for plotting
    # u_real = np.concatenate((u_train.detach().cpu().numpy(), u_test.detach().cpu().numpy()), axis=1)

    # plot_results_separate(u_real, u_train_NN, u_test_NN, dim=3, plot_type='Simulation')
    return heat.parameters()


def create_grid(start, end, step):
    x = torch.arange(start, end + step, step, device=device)
    y = torch.arange(start, end + step, step, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    grid = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), torch.zeros((grid_x.numel(), 1), device=device)],
                     dim=1)
    x, y = grid[:, 0], grid[:, 1]
    

    u0 = torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)/0.8)
    plate_length = int(np.sqrt(grid.shape[0]))
    u0_reshaped = u0.reshape(plate_length, plate_length)

    return grid, u0_reshaped



if __name__ == '__main__':
    num_epochs = 1000
    hidden_dim = 1000
    learning_rate = 1e-2
    reg = 'l1'
    
    for i in range(1,5):
        
        # params = train(i)
          
        # #plot params
        # plt.imshow(list(params)[0].detach().cpu().numpy(), origin='lower')
        # plt.colorbar()
        # plt.savefig(f'Experiments/smooth/alpha_normal_{round(i*0.1,2)}.png', dpi = 300)
        
        params = train_hybrid(i)
        plt.imshow(list(params)[0].detach().cpu().numpy(), origin='lower')
        plt.colorbar()
        plt.savefig(f'Experiments/smooth/alpha_hybrid_{round(i*0.1,2)}.png', dpi = 300)
        
        
        
        
    