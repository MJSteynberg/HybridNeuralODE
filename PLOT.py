# plot gaussians with given parameters

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os 
import matplotlib.patheffects as pe
from datetime import datetime
from data.dataloaders import DataLoader_Scalar

def moving_avg_std(loss, window=10):
    """
    Returns the moving average, standard deviation, and x-axis (epoch indices)
    for a given loss history array.
    """
    if len(loss) < window:
        # Not enough points for moving stats, so return original arrays.
        x = np.arange(len(loss))
        return x, loss, np.zeros_like(loss)
    avg = np.convolve(loss, np.ones(window)/window, mode='valid')
    std = np.array([np.std(loss[i:i+window]) for i in range(len(loss)-window+1)])
    x = np.arange(window-1, len(loss))
    return x, avg, std

def gaussian(param, num_gaussians=1, N=100, L=8):
    # interpolate alpha to the grid
    gaussian_map = 0.1*np.ones((N, N), dtype=np.float32)
    L = 8
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
    param_files = [f for f in files if f.startswith('param_') and f.endswith('.csv') and not f.startswith('param_pinn_')]
    param_pinn_files = [f for f in files if f.startswith('param_pinn_') and f.endswith('.csv')]
    index_files = [f for f in files if f.startswith('index_') and f.endswith('.csv')]
    error_files = [f for f in files if f.startswith('errors_') and f.endswith('.csv')]

    # Sort files based on the timestamp in the filename
    param_files.sort(key=lambda x: datetime.strptime(x[6:25], '%Y-%m-%d_%H-%M-%S'), reverse=True)
    param_pinn_files.sort(key=lambda x: datetime.strptime(x[11:30], '%Y-%m-%d_%H-%M-%S'), reverse=True)
    index_files.sort(key=lambda x: datetime.strptime(x[6:25], '%Y-%m-%d_%H-%M-%S'), reverse=True)
    error_files.sort(key=lambda x: datetime.strptime(x[7:26], '%Y-%m-%d_%H-%M-%S'), reverse=True)

    # Load the most recent param and index files
    recent_param_file = param_files[0]
    recent_param_pinn_file = param_pinn_files[0]
    recent_index_file = index_files[0]
    recent_error_file = error_files[0]

    # combine pinn param in param df


    param_df = pd.read_csv(os.path.join(folder, recent_param_file))
    param_pinn_df = pd.read_csv(os.path.join(folder, recent_param_pinn_file))
    param_df["params_pinn"] = param_pinn_df["params"]
    # rename params to params_pinn
    param_df = param_df.rename(columns={'params': 'params_pinn'})
    index_df = pd.read_csv(os.path.join(folder, recent_index_file))

    error_df = pd.read_csv(os.path.join(folder, recent_error_file))


    return param_df, index_df, error_df

def plot_gaussians():
    torch.set_rng_state(torch.manual_seed(42).get_state())
    device = 'cpu'
    results_folder = 'parameters/adv_diff/'
    data_folder = 'data/adv_diff'
    
    params, indices = load_files(results_folder)
    data = DataLoader_Scalar(device, data_folder)
    u_train = data.u[:, indices['training_indices'], :].detach().numpy()
    
    # Compute gaussian maps for each model.
    # We now include a PINN version.
    L_extent = 4  # image extent (from -4 to 4)
    # For alpha maps, lower-index elements and for kappa maps, later ones.
    alpha_phys  = gaussian(params["params_phys"][:4].values)
    alpha_pinn  = gaussian(params["params_pinn"][:4].values)
    alpha_hybrid= gaussian(params["params_hybrid"][:4].values)
    alpha_real  = gaussian(params["params_real"][:4].values)
    
    kappa_phys  = gaussian(params["params_phys"][4:].values)
    kappa_pinn  = gaussian(params["params_pinn"][4:].values)
    kappa_hybrid= gaussian(params["params_hybrid"][4:].values)
    kappa_real  = gaussian(params["params_real"][4:].values)
    
    # Scale the color limits using the real solution only.
    vmin_kappa, vmax_kappa = kappa_real.min(), kappa_real.max()
    vmin_alpha, vmax_alpha = alpha_real.min(), alpha_real.max()

    # Define the gridspec: Four images per row plus a colorbar axis (total 5 columns)
    fig = plt.figure(figsize=(10, 4.5))
    gs = fig.add_gridspec(
        2, 5,
        width_ratios=[1, 1, 1, 1, 0.2],
        left=0.05, right=0.93, top=0.93, bottom=0.03,
        wspace=0.1, hspace=0.1
    )
    
    # Top row: Kappa maps in order: Physics, PINN, Hybrid, True.
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax_top_cb = fig.add_subplot(gs[0, 4])
    # Bottom row: Alpha maps in order: Physics, PINN, Hybrid, True.
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[1, 3])
    ax_bot_cb = fig.add_subplot(gs[1, 4])
    
    # Remove ticks and enforce square aspect ratio for all image axes.
    for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    
    # Plot top row (Kappa maps)
    cs0 = ax0.imshow(kappa_phys, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_kappa, vmax=vmax_kappa)
    ax0.set_title("FD")
    cs1 = ax1.imshow(kappa_pinn, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_kappa, vmax=vmax_kappa)
    ax1.set_title("PINN")
    cs2 = ax2.imshow(kappa_hybrid, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_kappa, vmax=vmax_kappa)
    ax2.set_title("HYCO")
    cs3 = ax3.imshow(kappa_real, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_kappa, vmax=vmax_kappa)
    ax3.set_title("True")
    cbar_top = fig.colorbar(cs0, cax=ax_top_cb, orientation="vertical", pad=0.02)
    cbar_top.ax.tick_params(labelsize=10)
    
    # Optionally, overlay training trajectories on the Physics (ax0) subplot.
    # Assumes u_train has shape (time, n_trajectories, dimensions) where dimensions>=2.
    for traj in range(u_train.shape[1]):
        ax0.plot(u_train[:, traj, 0], u_train[:, traj, 1], 'red', linewidth=1)
        ax1.plot(u_train[:, traj, 0], u_train[:, traj, 1], 'red', linewidth=1)
        ax2.plot(u_train[:, traj, 0], u_train[:, traj, 1], 'red', linewidth=1)
        ax4.plot(u_train[:, traj, 0], u_train[:, traj, 1], 'red', linewidth=1)
        ax5.plot(u_train[:, traj, 0], u_train[:, traj, 1], 'red', linewidth=1)
        ax6.plot(u_train[:, traj, 0], u_train[:, traj, 1], 'red', linewidth=1)

    
    # Plot bottom row (Alpha maps)
    cs4 = ax4.imshow(alpha_phys, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_alpha, vmax=vmax_alpha)
    cs5 = ax5.imshow(alpha_pinn, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_alpha, vmax=vmax_alpha)
    cs6 = ax6.imshow(alpha_hybrid, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_alpha, vmax=vmax_alpha)
    cs7 = ax7.imshow(alpha_real, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_alpha, vmax=vmax_alpha)
    cbar_bot = fig.colorbar(cs4, cax=ax_bot_cb, orientation="vertical", pad=0.02)
    cbar_bot.ax.tick_params(labelsize=10)
    
    # Add overall row labels.
    fig.text(0.03, 0.75, r"$\kappa$", rotation=0, ha="center", va="center", fontsize=16)
    fig.text(0.03, 0.25, r"$\eta$", rotation=0, ha="center", va="center", fontsize=16)
    
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
    
def plot_error():
    """
    Plots the error between the predicted and true values of alpha and center.
    """
    results_folder = 'parameters/adv_diff/'
    params, indices, errors = load_files(results_folder)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the error
    ax.plot(errors['error_hybrid_syn'], label='Hybrid', color='blue')
    ax.plot(errors['error_hybrid_phys'], label='Hybrid Physics', color='green')
    ax.plot(errors['error_phys'], label='Physics', color='red')

    # Set labels and title
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Error', fontsize=14)
    ax.set_title('Error Comparison: Hybrid vs Physics', fontsize=16)
    ax.set_yscale('log')
    # Add grid and legend
    ax.grid(True)
    ax.legend()

    # Save the plot
    plt.savefig(f'{results_folder}error_comparison.png', dpi=500)

def plot_gaussians_with_error():
    torch.set_rng_state(torch.manual_seed(42).get_state())
    device = 'cpu'
    results_folder = 'parameters/adv_diff/'
    data_folder = 'data/adv_diff'
    
    # load files including errors
    params, indices, errors = load_files(results_folder)
    data = DataLoader_Scalar(device, data_folder)
    u_train = data.u[:, indices['training_indices'], :].detach().numpy()
    
    # Compute gaussian maps for each model.
    # For alpha maps, lower-index elements and for kappa maps, later ones.
    L_extent = 4  # image extent (from -4 to 4)
    alpha_phys   = gaussian(params["params_phys"][:4].values)
    alpha_pinn   = gaussian(params["params_pinn"][:4].values)
    alpha_hybrid = gaussian(params["params_hybrid"][:4].values)
    alpha_real   = gaussian(params["params_real"][:4].values)
    
    kappa_phys   = gaussian(params["params_phys"][4:].values)
    kappa_pinn   = gaussian(params["params_pinn"][4:].values)
    kappa_hybrid = gaussian(params["params_hybrid"][4:].values)
    kappa_real   = gaussian(params["params_real"][4:].values)
    
    # Color limits for images (scaled with true)
    vmin_kappa, vmax_kappa = kappa_real.min(), kappa_real.max()
    vmin_alpha, vmax_alpha = alpha_real.min(), alpha_real.max()
    
    # Create a gridspec with 3 rows and 5 columns.
    # The first two rows will hold the Gaussian maps, the bottom row the error plot.
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(
        3, 5,
        width_ratios=[1, 1, 1, 1, 0.2],
        height_ratios=[1, 1, 0.8],
        left=0.05, right=0.93, top=0.93, bottom=0.05,
        wspace=0.1, hspace=0.15
    )
    
    # --- Top row: Kappa maps ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax_top_cb = fig.add_subplot(gs[0, 4])
    
    cs0 = ax0.imshow(kappa_phys, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_kappa, vmax=vmax_kappa)
    ax0.set_title("FD")
    cs1 = ax1.imshow(kappa_pinn, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_kappa, vmax=vmax_kappa)
    ax1.set_title("PINN")
    cs2 = ax2.imshow(kappa_hybrid, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_kappa, vmax=vmax_kappa)
    ax2.set_title("HYCO")
    cs3 = ax3.imshow(kappa_real, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_kappa, vmax=vmax_kappa)
    ax3.set_title("True")
    cbar_top = fig.colorbar(cs0, cax=ax_top_cb, orientation="vertical", pad=0.02)
    cbar_top.ax.tick_params(labelsize=10)
    
    # --- Middle row: Alpha maps ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[1, 3])
    ax_bot_cb = fig.add_subplot(gs[1, 4])
    
    cs4 = ax4.imshow(alpha_phys, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_alpha, vmax=vmax_alpha)
    ax4.set_title("FD")
    cs5 = ax5.imshow(alpha_pinn, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_alpha, vmax=vmax_alpha)
    ax5.set_title("PINN")
    cs6 = ax6.imshow(alpha_hybrid, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_alpha, vmax=vmax_alpha)
    ax6.set_title("HYCO")
    cs7 = ax7.imshow(alpha_real, extent=(-L_extent, L_extent, -L_extent, L_extent),
                       origin='lower', cmap='viridis', vmin=vmin_alpha, vmax=vmax_alpha)
    ax7.set_title("True")
    cbar_bot = fig.colorbar(cs4, cax=ax_bot_cb, orientation="vertical", pad=0.02)
    cbar_bot.ax.tick_params(labelsize=10)
    
    # Remove ticks and enforce square aspect ratio for the image panels.
    for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
    # Optionally overlay training trajectories on selected panels.
    for ax in [ax0, ax4]:
        for traj in range(u_train.shape[1]):
            ax.plot(u_train[:, traj, 0], u_train[:, traj, 1], 'red', linewidth=1)
    
    # --- Bottom row: Error evolution plot ---
    # Create an axis that spans all 4 columns (we won't use column 4 here)
    ax_error = fig.add_subplot(gs[2, :4])
    
    # Plot the error curves (assuming errors contains keys as in your error plot code)
    ax_error.plot(errors['error_hybrid_syn'], label='Hybrid', color='blue')
    ax_error.plot(errors['error_hybrid_phys'], label='Hybrid Physics', color='green')
    ax_error.plot(errors['error_phys'], label='Physics', color='red')
    
    ax_error.set_xlabel('Epochs', fontsize=14)
    ax_error.set_ylabel('Error', fontsize=14)
    ax_error.set_title('Error Comparison: Hybrid vs Physics', fontsize=16)
    ax_error.set_yscale('log')
    ax_error.grid(True)
    ax_error.legend()
    
    plt.savefig(f'{results_folder}adv_diff_with_error.png', dpi=500)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_final_evolution_no_difference():
    """
    Creates a final evolution plot showing the final predictions for FD, PINN, HYCO, and True 
    for both kappa and η, along with the loss history below.
    """
    results_folder = 'parameters/adv_diff/'
    # Load parameters and errors (assumes errors includes loss histories)
    params, indices, errors = load_files(results_folder)
    
    # In this example, assume kappa data are stored in elements [4:],
    # and η (or alpha) in elements [:4]
    final_phy   = gaussian(params["params_phys"][4:].values)    # FD / Physics
    final_pinn  = gaussian(params["params_pinn"][4:].values)      # PINN
    final_hyb   = gaussian(params["params_hybrid"][4:].values)    # HYCO
    final_true  = gaussian(params["params_real"][4:].values)      # True
    
    eta_phy   = gaussian(params["params_phys"][:4].values)
    eta_pinn  = gaussian(params["params_pinn"][:4].values)
    eta_hyb   = gaussian(params["params_hybrid"][:4].values)
    eta_true  = gaussian(params["params_real"][:4].values)
    
    # Build a grid over the spatial domain.
    L_extent = 4
    N_x = final_true.shape[0]
    N_y = final_true.shape[1]
    x = np.linspace(-L_extent, L_extent, N_x)
    y = np.linspace(-L_extent, L_extent, N_y)
    xx, yy = np.meshgrid(x, y)
    
    # Set a common range for the predictions.
    vmin = np.floor( min(final_true.min(), eta_true.min()) )
    vmax = np.ceil( max(final_true.max(), eta_true.max()) )
    
    fig = plt.figure(figsize=(12, 12))
    # Create a gridspec for 2 rows and 5 columns (4 panels + 1 for colorbar)
    gs_top = fig.add_gridspec(
        2, 5, 
        width_ratios=[1, 1, 1, 1, 0.2],
        left=0.07, right=0.93, top=0.93, bottom=0.45,
        wspace=0.1, hspace=0.1
    )
    
    # --- Row 1: Kappa maps ---
    ax0 = fig.add_subplot(gs_top[0, 0])
    cf0 = ax0.contourf(xx, yy, final_phy, levels=100, vmin=vmin, vmax=vmax, cmap="viridis")
    ax0.set_xticks([]); ax0.set_yticks([])
    
    ax1 = fig.add_subplot(gs_top[0, 1])
    cf1 = ax1.contourf(xx, yy, final_pinn, levels=100, vmin=vmin, vmax=vmax, cmap="viridis")
    ax1.set_xticks([]); ax1.set_yticks([])
    
    ax2 = fig.add_subplot(gs_top[0, 2])
    cf2 = ax2.contourf(xx, yy, final_hyb, levels=100, vmin=vmin, vmax=vmax, cmap="viridis")
    ax2.set_xticks([]); ax2.set_yticks([])
    
    ax3 = fig.add_subplot(gs_top[0, 3])
    cf3 = ax3.contourf(xx, yy, final_true, levels=100, vmin=vmin, vmax=vmax, cmap="viridis")
    ax3.set_xticks([]); ax3.set_yticks([])
    
    # Unified colorbar for row 1 (Kappa)
    ax_cb_top = fig.add_subplot(gs_top[0, 4])
    norm = Normalize(vmin=vmin, vmax=vmax)
    mappable = ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array([])
    cb_top = fig.colorbar(mappable, cax=ax_cb_top)
    cb_top.set_label("Parameter Value", fontsize=14)
    
    # --- Row 2: η maps ---
    ax4 = fig.add_subplot(gs_top[1, 0])
    cf4 = ax4.contourf(xx, yy, eta_phy, levels=100, vmin=vmin, vmax=vmax, cmap="viridis")
    ax4.set_xticks([]); ax4.set_yticks([])
    
    ax5 = fig.add_subplot(gs_top[1, 1])
    cf5 = ax5.contourf(xx, yy, eta_pinn, levels=100, vmin=vmin, vmax=vmax, cmap="viridis")
    ax5.set_xticks([]); ax5.set_yticks([])
    
    ax6 = fig.add_subplot(gs_top[1, 2])
    cf6 = ax6.contourf(xx, yy, eta_hyb, levels=100, vmin=vmin, vmax=vmax, cmap="viridis")
    ax6.set_xticks([]); ax6.set_yticks([])
    
    ax7 = fig.add_subplot(gs_top[1, 3])
    cf7 = ax7.contourf(xx, yy, eta_true, levels=100, vmin=vmin, vmax=vmax, cmap="viridis")
    ax7.set_xticks([]); ax7.set_yticks([])
    
    # Unified colorbar for row 2 (η)
    ax_cb_bot = fig.add_subplot(gs_top[1, 4])
    mappable2 = ScalarMappable(norm=norm, cmap="viridis")
    mappable2.set_array([])
    cb_bot = fig.colorbar(mappable2, cax=ax_cb_bot)
    cb_bot.set_label("Parameter Value", fontsize=14)
    
    # --- Add column labels (above the grid) ---
    col_labels = ["FD", "PINN", "HYCO", "True"]
    col_centers = [0.07 + 0.86*(i+0.5)/4 for i in range(4)]
    for label, xc in zip(col_labels, col_centers):
        fig.text(xc, 0.96, label, ha="center", fontsize=18)
    
    # --- Add row labels (to the left of the grid) ---
    fig.text(0.03, 0.77, r"$\kappa$", ha="center", fontsize=22)
    fig.text(0.03, 0.32, r"$\eta$", ha="center", fontsize=22)
    
    # --- Loss history subplot using sliding window average ---
    window = 100
    # Compute moving average and standard deviation for each loss history.
    # (Assumes moving_avg_std is defined and takes an array and window size as inputs.)
    x_phys, avg_phys, std_phys = moving_avg_std(errors['error_phys'], window=window)
    x_hyb, avg_hyb, std_hyb = moving_avg_std(errors['error_hybrid_phys'], window=window)
    x_synth, avg_synth, std_synth = moving_avg_std(errors['error_hybrid_syn'], window=window)

    ax_loss = fig.add_axes([0.07, 0.07, 0.86, 0.3])
    ax_loss.plot(x_phys, avg_phys, label="FEM Error", c=u'#1f77b4')
    ax_loss.fill_between(x_phys, avg_phys - std_phys, avg_phys + std_phys, alpha=0.3, color=u'#1f77b4')

    ax_loss.plot(x_hyb, avg_hyb, label="HYCO Physical Error", linestyle="--", c = u'#2ca02c')
    ax_loss.fill_between(x_hyb, avg_hyb - std_hyb, avg_hyb + std_hyb, alpha=0.3, color = u'#2ca02c')

    ax_loss.plot(x_synth, avg_synth, label="HYCO Synthetic Error", linestyle="-.", c = u'#d62728')
    ax_loss.fill_between(x_synth, avg_synth - std_synth, avg_synth + std_synth, alpha=0.3, color = u'#d62728')

    #  u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728'
    # Set y-axis limits to not be too far from the average.
    avg_min = min(avg_phys.min(), avg_hyb.min(), avg_synth.min())
    avg_max = max(avg_phys.max(), avg_hyb.max(), avg_synth.max())
    ax_loss.set_ylim(avg_min * 0.8, avg_max * 1.4)

    ax_loss.set_yscale("log")
    ax_loss.set_title("Mean Squared Error History", fontsize=16)
    ax_loss.set_xlabel("Epoch", fontsize=16)
    ax_loss.set_ylabel("Mean Squared Error", fontsize=16)
    ax_loss.legend(fontsize=12)
        
    filename = f"{results_folder}final_evolution_no_diff.png"
    plt.savefig(filename, dpi=500)
    plt.close(fig)

plot_final_evolution_no_difference()