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
from models.FEM import AdvectionDiffusionG
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda')


def train_hybrid(i):
    # =============================================================================
    # DATA
    # =============================================================================
    u = np.load('Experiments/gaussians/data.npy')
    u = torch.tensor(u, dtype=torch.float32)
    t = torch.linspace(0, 1, 100)
    
    length_u = u.shape[1]
    
    torch.set_rng_state(torch.manual_seed(42).get_state())
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(0, u.shape[1]):
    #     ax.plot(u[:, i, 0].detach().cpu().numpy(), u[:, i, 1].detach().cpu().numpy(), u[:, i, 2].detach().cpu().numpy())
    # plt.show()

    # Create indices and split for train and test data
    train_size = int(0.2 * length_u)
    print(train_size)
    indices = torch.randperm(length_u)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    u_train = u[:, train_indices, :].to(device)
    u_test = u[:, test_indices, :].to(device)
    
    nX = 21
    # Define a grid of points within [-3, 3] with an added column of zeros
    grid, u0 = create_grid(-3, 3, 6/(nX-1))

    # =============================================================================
    # SETUP
    # =============================================================================
    T, num_steps = 1, 100
    dt = T / (num_steps - 1)
    data_dim = u.shape[2]

    # Choose optimizer settings
    L1 = False
    weight_decay = 0.0 if L1 else 0.01 * dt
    num_gaussians_alpha = 1
    num_gaussians_kappa = 1
    L = 6
    N = nX
    # =============================================================================
    # MODEL SETUP
    # =============================================================================
    anode = NeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='relu',
                      architecture='bottleneck', T=T, time_steps=num_steps).to(device)

    num_gaussians_alpha = 1
    num_gaussians_kappa = 1
    alpha = torch.tensor([1, 1.4, -1.3, 1.0]).float() # [Amplitude, Amplitude, x0, x0, y0, y0, sigma, sigma]
    kappa = torch.tensor([1, -2, -1.1, 1.0]).float().to(device) # [Amplitude, Amplitude, x0, x0, y0, y0, sigma, sigma]
    heat = AdvectionDiffusionG(device, L, N, dt, num_steps, num_gaussians_alpha, num_gaussians_kappa, alpha = alpha, kappa = kappa)
    heat.create_advection_map()
    heat.create_diffusion_map()
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
    torch.save(anode.state_dict(), f'Experiments/gaussians/anode_{i}.pt')
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
    u = np.load('Experiments/gaussians/data.npy')
    u = torch.tensor(u, dtype=torch.float32)
    t = torch.linspace(0, 1, 100)
    
    length_u = u.shape[1]
    
    torch.set_rng_state(torch.manual_seed(42).get_state())
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(0, u.shape[1]):
    #     ax.plot(u[:, i, 0].detach().cpu().numpy(), u[:, i, 1].detach().cpu().numpy(), u[:, i, 2].detach().cpu().numpy())
    # plt.show()

    # Create indices and split for train and test data
    train_size = int(0.2 * length_u)
    print(train_size)
    indices = torch.randperm(length_u)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    u_train = u[:, train_indices, :].to(device)
    u_test = u[:, test_indices, :].to(device)
    
    nX = 21
    # Define a grid of points within [-3, 3] with an added column of zeros
    grid, u0 = create_grid(-3, 3, 6/(nX-1))

    # =============================================================================
    # SETUP
    # =============================================================================
    T, num_steps = 1, 100
    dt = T / (num_steps - 1)
    data_dim = u.shape[2]

    # Choose optimizer settings
    L1 = True
    weight_decay = 0.0 if L1 else 0.01 * dt
    num_gaussians_alpha = 1
    num_gaussians_kappa = 1
    L = 6
    N = nX
    # =============================================================================
    # MODEL SETUP
    # =============================================================================
    anode = NeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='relu',
                      architecture='bottleneck', T=T, time_steps=num_steps).to(device)

    num_gaussians_alpha = 1
    num_gaussians_kappa = 1
    alpha = torch.tensor([1, 1.4, -1.3, 1.0]).float() # [Amplitude, Amplitude, x0, x0, y0, y0, sigma, sigma]
    kappa = torch.tensor([1, -2, -1.1, 1.0]).float().to(device) # [Amplitude, Amplitude, x0, x0, y0, y0, sigma, sigma]
    heat = AdvectionDiffusionG(device, L, N, dt, num_steps, num_gaussians_alpha, num_gaussians_kappa, alpha = alpha, kappa = kappa)

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
    
    L = 6
    u0 = - torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)) + torch.exp(-((x - 1) ** 2 + (y - 1) ** 2))
    plate_length = int(np.sqrt(grid.shape[0]))
    u0_reshaped = u0.reshape(plate_length, plate_length)

    return grid, u0_reshaped



if __name__ == '__main__':
    num_epochs = 2000
    hidden_dim = 1000
    learning_rate = 1e-3
    reg = 'l1'
    
    params = train_hybrid(0)
    print(params)
    
    params = train(0)
    print(params)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    
    
    
        
        
        
        
        
    # for i in range(1,5):
    #     alpha_hybrid = np.load(f'Experiments/2materials/alpha_h_{i}.npy')
    #     alpha = np.load(f'Experiments/2materials/alpha_{i}.npy')
    #     u_train = np.load(f'Experiments/2materials/u_train_{i}.npy')
    #     u_train_hybrid = np.load(f'Experiments/2materials/u_train_h_{i}.npy')
        
    #     fig = plt.figure()
        
    #     grid, u0 = create_grid(-3, 3, 0.6)
        
    #     print(grid.shape)
        
    #     grid = grid.reshape(11, 11, 3)
    #     print(grid[1,1,0], grid[1,1,1])
    #     print(grid[1,-2,0], grid[1,-2,1])
    #     print(grid[-2,1,0], grid[-2,1,1])
    #     print(grid[-2,-2,0], grid[-2,-2,1])
        
        
    #     ax1 = fig.add_subplot(121)
    #     ax2 = fig.add_subplot(122)
    #     ax1.set_xlim(-3, 3)
    #     ax1.set_ylim(-3, 3)
    #     ax2.set_xlim(-3, 3)
    #     ax2.set_ylim(-3, 3)
    #     ax1.imshow(alpha[-1].T, extent=(-3, 3, -3, 3), origin='lower', interpolation='bilinear', cmap='coolwarm')
    #     ax2.imshow(alpha_hybrid[-1].T, extent=(-3, 3, -3, 3), origin='lower', interpolation='bilinear', cmap='coolwarm')
    #     # plot the value at the center of each of the four corners
    #     ax1.text(-2, -2, round(alpha[-1][1, 1],2), ha='center', va='center', color='black')
    #     ax1.text(-2, 2, round(alpha[-1][1, -2],2), ha='center', va='center', color='black')
    #     ax1.text(2, -2, round(alpha[-1][-2, 1],2), ha='center', va='center', color='black')
    #     ax1.text(2, 2, round(alpha[-1][-2, -2],2), ha='center', va='center', color='black')
    #     ax1.set_title('Finite Difference')
        
    #     ax2.text(-2, -2, round(alpha_hybrid[-1][1, 1],2), ha='center', va='center', color='black')
    #     ax2.text(-2, 2, round(alpha_hybrid[-1][1, -2],2), ha='center', va='center', color='black')
    #     ax2.text(2, -2, round(alpha_hybrid[-1][-2, 1],2), ha='center', va='center', color='black')
    #     ax2.text(2, 2, round(alpha_hybrid[-1][-2, -2],2), ha='center', va='center', color='black')
    #     ax2.set_title('Hybrid')
        
    #     plt.savefig(f'Experiments/2materials/alpha_{round(i*0.1,2)}_no_traj.png', dpi = 300)
        
    #     # plot the (xy) trajectories of the heat source
    #     print(u_train.shape)
    #     for j in range(0, u_train.shape[1]):
    #         ax1.plot(u_train[:, j, 0], u_train[:, j, 1], c='k')
    #     for j in range(0, u_train_hybrid.shape[1]):
    #         ax2.plot(u_train_hybrid[:, j, 0], u_train_hybrid[:, j, 1], c='k')

        
    #     plt.savefig(f'Experiments/2materials/alpha_{round(i*0.1,2)}.png', dpi = 300)
        
            

        

        
        
        