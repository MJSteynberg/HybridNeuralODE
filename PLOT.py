# plot gaussians with given parameters

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os 
from datetime import datetime
from data.dataloaders import DataLoader_Scalar

def gaussian(param, num_gaussians=1, N=100, L=6):
    # interpolate alpha to the grid
    gaussian_map = 0.1*np.ones((N, N), dtype=np.float32)
    
    x = np.linspace(-L//2, L//2, N)
    y = np.linspace(-L//2, L//2, N)
    x, y = np.meshgrid(x, y, indexing='ij')
    for i in range(num_gaussians):
        gaussian_map += param[i] * np.exp(-((x - param[num_gaussians + i]) ** 2 + (y - param[2*num_gaussians + i]) ** 2))
    return gaussian_map

def load_files(folder):

    # List all files in the directory
    files = os.listdir(folder)

    # Filter files that match the naming pattern 'param_YYYY-MM-DD_HH-MM-SS.csv'
    param_files = [f for f in files if f.startswith('param_') and f.endswith('.csv')]
    index_files = [f for f in files if f.startswith('index_') and f.endswith('.csv')]

    # Sort files based on the timestamp in the filename
    param_files.sort(key=lambda x: datetime.strptime(x[6:25], '%Y-%m-%d_%H-%M-%S'), reverse=True)
    index_files.sort(key=lambda x: datetime.strptime(x[6:25], '%Y-%m-%d_%H-%M-%S'), reverse=True)

    # Load the most recent param and index files
    recent_param_file = param_files[0]
    recent_index_file = index_files[0]

    param_df = pd.read_csv(os.path.join(folder, recent_param_file))
    index_df = pd.read_csv(os.path.join(folder, recent_index_file))

    return param_df, index_df

def plot_gaussians():
    torch.set_rng_state(torch.manual_seed(42).get_state())
    device = 'cpu'

    results_folder = 'parameters/adv_diff/'
    data_folder = 'data/adv_diff'

    params, indices = load_files(results_folder)
    data = DataLoader_Scalar(device, data_folder)
    u_train = data.u[:, indices['training_indices'], :].detach().numpy()
    


    alpha_real = gaussian(params["params_real"][:4].values)
    alpha_hybrid = gaussian(params["params_hybrid"][:4].values)
    alpha_phys = gaussian(params["params_phys"][:4].values)
    kappa_real = gaussian(params["params_real"][4:].values)
    kappa_hybrid = gaussian(params["params_hybrid"][4:].values)
    kappa_phys = gaussian(params["params_phys"][4:].values)
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    colormap = 'viridis'

    L = 3
    global_max = np.max(np.concatenate((alpha_real, kappa_real)))

    # Setup all heatmaps
    im1 = axs[1][0].imshow(alpha_phys,
                        extent=(-L, L, -L, L),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max - 0.5)
    im2 = axs[1][1].imshow(alpha_hybrid,
                        extent=(-L, L, -L, L),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max - 0.5)
    im3 = axs[1][2].imshow(alpha_real,
                        extent=(-L, L, -L, L),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max - 0.5)

    im1 = axs[0][0].imshow(kappa_phys,
                        extent=(-L, L, -L, L),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max - 0.5)
    im2 = axs[0][1].imshow(kappa_hybrid,
                        extent=(-L, L, -L, L),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max - 0.5)
    im3 = axs[0][2].imshow(kappa_real,
                        extent=(-L, L, -L, L),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max - 0.5)

    # Calculate L1 errors
    alpha_mean_error_physics = abs(alpha_phys - alpha_real).mean()
    alpha_mean_error_hybrid = abs(alpha_hybrid - alpha_real).mean()
    c_mean_error_physics = abs(kappa_phys - kappa_real).mean()
    c_mean_error_hybrid = abs(kappa_hybrid - kappa_real).mean()


    # Set titles including errors
    plt.rcParams['axes.titlesize'] = 14
    axs[1][0].set_title(r"$\kappa(x)   $ Physics: " + "\n" +
                        r"Mean Error:  $%.3e$" % alpha_mean_error_physics)
    axs[1][1].set_title(r"$\kappa(x)   $ Hybrid:" + "\n" +
                        r"Mean Error:  $%.3e$" % alpha_mean_error_hybrid)
    axs[1][2].set_title(r"$\kappa(x)  $ Real")

    axs[0][0].set_title(r"$\eta(x)  $ Physics:"+ "\n" +
                        r"Mean Error:  $%.3e$" % c_mean_error_physics)
    axs[0][1].set_title(r"$\eta(x)   $ Hybrid:"+ "\n" +
                        r"Mean Error:  $%.3e$" % c_mean_error_hybrid)
    axs[0][2].set_title(r"$\eta(x)  $ Real")

    # Set axis limits
    for i in axs:
        for ax in i:
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)

    # Plot training data
    for j in range(u_train.shape[1]):
        axs[0][0].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')
        axs[0][1].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')
        axs[1][0].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')
        axs[1][1].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')

    fig.colorbar(im3, ax=axs, orientation='vertical', shrink=0.8, label="Legend")
    plt.savefig(f'{results_folder}adv_diff.png', dpi=500)

def plot_error():

    torch.set_rng_state(torch.manual_seed(42).get_state())
    device = 'cpu'

    results_folder = 'parameters/adv_diff/'
    data_folder = 'data/adv_diff'

    params, indices = load_files(results_folder)
    data = DataLoader_Scalar(device, data_folder)
    u_train = data.u[:, indices['training_indices'], :].detach().numpy()
    


    alpha_real = gaussian(params["params_real"][:4].values)
    alpha_hybrid = gaussian(params["params_hybrid"][:4].values)
    alpha_phys = gaussian(params["params_phys"][:4].values)
    kappa_real = gaussian(params["params_real"][4:].values)
    kappa_hybrid = gaussian(params["params_hybrid"][4:].values)
    kappa_phys = gaussian(params["params_phys"][4:].values)

    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colormap = 'viridis'
    L = 3
    global_max = max(np.max(np.abs(kappa_phys - kappa_real).flatten()), np.max(np.abs(kappa_hybrid - kappa_real).flatten()), np.max(np.abs(alpha_phys - alpha_real).flatten()), np.max(np.abs(alpha_hybrid - alpha_real).flatten()))

    # Setup heatmaps
    im1 = axs[1][0].imshow(np.abs(alpha_phys - alpha_real),
                        extent=(-L, L, -L, L),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max)
    im2 = axs[1][1].imshow(np.abs(alpha_hybrid - alpha_real),
                        extent=(-L, L, -L, L),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max)

    im1 = axs[0][0].imshow(np.abs(kappa_phys - kappa_real),
                        extent=(-L, L, -L, L),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max)
    im2 = axs[0][1].imshow(np.abs(kappa_hybrid - kappa_real),
                        extent=(-L, L, -L, L),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max)

    # Calculate L1 errors
    alpha_mean_error_physics = abs(alpha_phys - alpha_real).mean()
    alpha_mean_error_hybrid = abs(alpha_hybrid - alpha_real).mean()
    c_mean_error_physics = abs(kappa_phys - kappa_real).mean()
    c_mean_error_hybrid = abs(kappa_hybrid - kappa_real).mean()


    # Set titles including errors
    plt.rcParams['axes.titlesize'] = 14
    axs[1][0].set_title(r"$\kappa(x)   $Error Physics: " + "\n" +
                        r"Mean Error:  $%.3e$" % alpha_mean_error_physics)
    axs[1][1].set_title(r"$\kappa(x)   $Error Hybrid:" + "\n" +
                        r"Mean Error:  $%.3e$" % alpha_mean_error_hybrid)

    axs[0][0].set_title(r"$\eta(x)  $ Error Physics:"+ "\n" +
                        r"Mean Error:  $%.3e$" % c_mean_error_physics)
    axs[0][1].set_title(r"$\eta(x)   $ Error Hybrid:"+ "\n" +
                        r"Mean Error:  $%.3e$" % c_mean_error_hybrid)

    # Set axis limits
    for i in axs:
        for ax in i:
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)

    # Plot training data
    for j in range(u_train.shape[1]):
        axs[0][0].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')
        axs[0][1].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')
        axs[1][0].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')
        axs[1][1].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')

    fig.colorbar(im2, ax=axs, orientation='vertical', shrink=0.8, label="Legend")
    plt.savefig(f'{results_folder}adv_diff_error.png', dpi=500)





def compare_predictions():
    """
    Compares the predictions of alpha (physics, hybrid, real) and centers (physics, hybrid, real) on the same plot.
    """
    # Reshape alpha and center values into 3x2 and 3x4 matrices
    
    

    results_folder = 'parameters/adv_diff/'

    params, indices = load_files(results_folder)
    alpha_real = np.array([params['params_real'][0], params['params_real'][4]])
    alpha_hybrid = np.array([params['params_hybrid'][0], params['params_hybrid'][4]])
    alpha_phys = np.array([params['params_phys'][0], params['params_phys'][4]])
    alpha_values = np.array([alpha_real, alpha_hybrid, alpha_phys])

    centers_real = np.array(np.concatenate((params['params_real'][1:3], params['params_real'][5:7])))
    centers_hybrid = np.array(np.concatenate((params['params_hybrid'][1:3], params['params_hybrid'][5:7])))
    centers_phys = np.array(np.concatenate((params['params_phys'][1:3], params['params_phys'][5:7])))
    centers_values = np.array([centers_real.ravel(), centers_hybrid.ravel(), centers_phys.ravel()])
    
    

    # X-axis labels for 6 parameters
    parameter_names = [
        r'$\alpha_1$', r'$\alpha_2$', r'$c_{1,x}$', r'$c_{1,y}$', r'$c_{2,x}$', r'$c_{2,y}$'
    ]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the x-axis positions for each parameter
    x_positions = np.arange(6)

    # Enable the grid
    # Set zorder of grid to be below the points
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=14)  # Increase font size for both x and y axis ticks

    # Plot the alpha and centers values
    for i in range(3):
        # Set the marker and size for each scatter plot
        marker = 'o'  # Default is a circle
        if i == 0:  # For 'Real' we use a star
            marker = '*'
            size = 400  # Increased size for Real (stars)
        elif i == 2:
            marker = 'X'
            size = 150  # Default size for Hybrid and Physics
        else: 
            size = 150  # Default size for Hybrid and Physics

        # Plot alpha values (Red for Real, Green for Hybrid, Blue for Physics)
        ax.scatter(x_positions[:2], alpha_values[i, :], c='r' if i == 0 else 'b' if i == 1 else 'g',
                   label=r'$\alpha_{1,2}$ (' + ["Real", "Hybrid", "Physics"][i] + ')', s=size, edgecolors='k', marker=marker)

        # Plot center values (Green for Real, Blue for Hybrid, Red for Physics)
        ax.scatter(x_positions[2:], centers_values[i, :], c='r' if i == 0 else 'b' if i == 1 else 'g',
                   label=f'Centers ({["Real", "Hybrid", "Physics"][i]})', s=size, edgecolors='k', marker=marker)

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(parameter_names)

    # Set y-axis label
    ax.set_ylabel('Predicted Values', fontsize=14)

    # Add title
    ax.set_title(r'Comparison of Predicted $\alpha$ and Centers (Real vs Hybrid vs Physics)', fontsize=14)

    # Add a legend
    ax.legend()

    # Save the plot in the usual folder
    plt.savefig(f'{results_folder}compare_predictions.png', dpi=500)
    


compare_predictions()
plot_gaussians()
plot_error()

