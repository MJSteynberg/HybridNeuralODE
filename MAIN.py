#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Code
"""

from data.dataloaders import Dissipative
import torch
from torch.utils.data import DataLoader, Subset
from models.training import Trainer
from models.neural_odes import NeuralODE
from models.plot import plot_results_separate
from models.FEM import heatsolve
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # =============================================================================
    # DATA
    # =============================================================================
    filename_data = './u.mat'
    datasets = Dissipative(filename_data)
    t, u = datasets[:]

    # Create indices and split for train and test data
    train_size = int(0.6 * datasets.length_u())
    indices = torch.randperm(datasets.length_u())
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    u_train = u[:, train_indices, :].to(device)
    u_test = u[:, test_indices, :].to(device)

    # Define a grid of points within [-3, 3] with an added column of zeros
    grid, u0 = create_grid(-3, 3, 0.6)

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

    heat = heatsolve(device, T=T, nX=11, time_steps=num_steps).to(device)

    # Optimizers
    optimizer_anode = torch.optim.Adam(anode.dynamics.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_heat = torch.optim.Adam(heat.parameters(), lr=0.05)

    # =============================================================================
    # TRAINING
    # =============================================================================
    trainer_anode = Trainer(anode, heat, optimizer_anode, optimizer_heat, device, reg=reg, grid=grid)
    trainer_anode.train(u_train, u0, num_epochs)

    # =============================================================================
    # TESTING
    # =============================================================================
    def test_prediction(data, initial_condition):
        pred, traj_pred = anode(initial_condition)
        pred_save = torch.cat((initial_condition.unsqueeze(0), traj_pred), dim=0)
        return pred_save.detach().cpu().numpy()

    u_test_NN = test_prediction(u_test[0, :, :], u_test[0, :, :])
    u_train_NN = test_prediction(u_train[0, :, :], u_train[0, :, :])

    # Convert and combine real values for plotting
    u_real = np.concatenate((u_train.detach().cpu().numpy(), u_test.detach().cpu().numpy()), axis=1)

    plot_results_separate(u_real, u_train_NN, u_test_NN, dim=3, plot_type='Simulation')
    return t


def create_grid(start, end, step):
    x = torch.arange(start, end + step, step, device=device)
    y = torch.arange(start, end + step, step, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    grid = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), torch.zeros((grid_x.numel(), 1), device=device)],
                     dim=1)
    x_grid, y_grid = grid[:, 0], grid[:, 1]
    u0 = 5 * torch.exp(-5 * ((x_grid + 1) ** 2 + (y_grid + 1) ** 2)) - 3 * torch.exp(
        -5 * ((x_grid - 1) ** 2 + (y_grid - 1) ** 2))
    plate_length = int(np.sqrt(grid.shape[0]))
    u0_reshaped = u0.reshape(plate_length, plate_length)

    return grid, u0_reshaped


if __name__ == '__main__':
    num_epochs = 1000
    hidden_dim = 500
    learning_rate = 1e-4
    reg = 'l1'
    t = main()
