# plot gaussians with given parameters

import numpy as np
import matplotlib.pyplot as plt
import torch
def plot_gaussians():


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
    u_train = u[:, train_indices, :].detach().numpy()
    u_test = u[:, test_indices, :]


    def gaussian(param, num_gaussians=1, N=100, L=6):
        # interpolate alpha to the grid
        advection_map = 0.1*np.ones((N, N), dtype=np.float32)
        
        x = np.linspace(-L//2, L//2, N)
        y = np.linspace(-L//2, L//2, N)
        x, y = np.meshgrid(x, y, indexing='ij')
        for i in range(num_gaussians):
            advection_map += param[i] * np.exp(-((x + param[num_gaussians + i]) ** 2 + (y + param[2*num_gaussians + i]) ** 2))
        return advection_map

    alpha_real = gaussian(param_1)
    alpha_hybrid = gaussian(param_hybrid_1, num_gaussians=1)
    alpha_physics = gaussian(param_fd_1, num_gaussians=1)
    c_real = gaussian(param_2)
    c_hybrid = gaussian(param_hybrid_2)
    c_physics = gaussian(param_fd_2)
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    colormap = 'viridis'
    L = 3
    global_max = 3

    im1 = axs[1][0].imshow(alpha_physics,
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

    im1 = axs[0][0].imshow(c_physics,
                        extent=(-L, L, -L, L),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max - 0.5)
    im2 = axs[0][1].imshow(c_hybrid,
                        extent=(-L, L, -L, L),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max - 0.5)
    im3 = axs[0][2].imshow(c_real,
                        extent=(-L, L, -L, L),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max - 0.5)

    # for j in range(u_train.shape[1]):
    #     axs[0].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')
    #     axs[1].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')
    alpha_mean_error_physics = abs(alpha_physics - alpha_real).mean()
    alpha_mean_error_hybrid = abs(alpha_hybrid - alpha_real).mean()
    c_mean_error_physics = abs(c_physics - c_real).mean()
    c_mean_error_hybrid = abs(c_hybrid - c_real).mean()



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

    for i in axs:
        for ax in i:
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)

    for j in range(u_train.shape[1]):
        axs[0][0].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')
        axs[0][1].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')
        axs[1][0].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')
        axs[1][1].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')

    fig.colorbar(im3, ax=axs, orientation='vertical', shrink=0.8, label="Legend")
    plt.savefig('adv_diff.png', dpi=500)

def plot_error():


    torch.set_rng_state(torch.manual_seed(42).get_state())



    def gaussian(param, num_gaussians=1, N=100, L=6):
        # interpolate alpha to the grid
        advection_map = 0.1*np.ones((N, N), dtype=np.float32)
        
        x = np.linspace(-L//2, L//2, N)
        y = np.linspace(-L//2, L//2, N)
        x, y = np.meshgrid(x, y, indexing='ij')
        for i in range(num_gaussians):
            advection_map += param[i] * np.exp(-((x + param[num_gaussians + i]) ** 2 + (y + param[2*num_gaussians + i]) ** 2))
        return advection_map

    alpha_real = gaussian(param_1)
    alpha_hybrid = gaussian(param_hybrid_1, num_gaussians=1)
    alpha_physics = gaussian(param_fd_1, num_gaussians=1)
    c_real = gaussian(param_2)
    c_hybrid = gaussian(param_hybrid_2)
    c_physics = gaussian(param_fd_2)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colormap = 'viridis'
    L = 3
    global_max = max(np.max(np.abs(c_physics - c_real).flatten()), np.max(np.abs(c_hybrid - c_real).flatten()), np.max(np.abs(alpha_physics - alpha_real).flatten()), np.max(np.abs(alpha_hybrid - alpha_real).flatten()))

    im1 = axs[1][0].imshow(np.abs(alpha_physics - alpha_real),
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

    im1 = axs[0][0].imshow(np.abs(c_physics - c_real),
                        extent=(-L, L, -L, L),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max)
    im2 = axs[0][1].imshow(np.abs(c_hybrid - c_real),
                        extent=(-L, L, -L, L),
                        origin='lower',
                        cmap=colormap,
                        vmin=0,
                        vmax=global_max)

    # for j in range(u_train.shape[1]):
    #     axs[0].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')
    #     axs[1].plot(u_train[:, j, 0], u_train[:, j, 1], c='k')
    alpha_mean_error_physics = abs(alpha_physics - alpha_real).mean()
    alpha_mean_error_hybrid = abs(alpha_hybrid - alpha_real).mean()
    c_mean_error_physics = abs(c_physics - c_real).mean()
    c_mean_error_hybrid = abs(c_hybrid - c_real).mean()



    plt.rcParams['axes.titlesize'] = 14
    axs[1][0].set_title(r"$\kappa(x)   $Error Physics: " + "\n" +
                        r"Mean Error:  $%.3e$" % alpha_mean_error_physics)
    axs[1][1].set_title(r"$\kappa(x)   $Error Hybrid:" + "\n" +
                        r"Mean Error:  $%.3e$" % alpha_mean_error_hybrid)

    axs[0][0].set_title(r"$\eta(x)  $ Error Physics:"+ "\n" +
                        r"Mean Error:  $%.3e$" % c_mean_error_physics)
    axs[0][1].set_title(r"$\eta(x)   $ Error Hybrid:"+ "\n" +
                        r"Mean Error:  $%.3e$" % c_mean_error_hybrid)

    for i in axs:
        for ax in i:
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)


    fig.colorbar(im2, ax=axs, orientation='vertical', shrink=0.8, label="Legend")
    plt.savefig('adv_diff_error.png', dpi=500)


def compare_predictions(alpha_physics, alpha_hybrid, alpha_real, centers_physics, centers_hybrid, centers_real):
    """
    Compares the predictions of alpha (physics, hybrid, real) and centers (physics, hybrid, real) on the same plot.
    """
    # Reshape alpha and center values into 3x2 and 3x4 matrices
    alpha_values = np.array([alpha_real, alpha_hybrid, alpha_physics])
    centers_values = np.array([centers_real.ravel(), centers_hybrid.ravel(), centers_physics.ravel()])

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
    plt.savefig('compare_predictions.png', dpi=500)
    

param_1 = [3,  1, 1, 1.0]
param_2 = [2.5, -2, -2, 1.0]
param_hybrid_1 = [1.1534, 0.7218, 1.3790, 1.0000]
param_hybrid_2 = [ 2.9477, -2.0192, -2.1021,  1.0000]
param_fd_1 = [ 0.3815,  1.9029, -2.9706,  1.0000]
param_fd_2 = [ 2.9477, -2.0193, -2.1021,  1.0000]

u = np.load('Experiments/gaussians/data.npy')
u = torch.tensor(u, dtype=torch.float32)
t = torch.linspace(0, 1, 100)

length_u = u.shape[1]

alpha_physics = np.array([0.3815, 2.9477])
centers_physics = np.array([1.9029, -2.9706, -2.0193, -2.1021])

alpha_hybrid = np.array([1.1534, 2.9477])
centers_hybrid = np.array([0.7218, 1.3790, -2.0192, -2.1021])

alpha_real = np.array([3, 2.5])
centers_real = np.array([1, 1, -2, -2])
compare_predictions(alpha_physics, alpha_hybrid, alpha_real, centers_physics, centers_hybrid, centers_real)

#plot_gaussians()

plot_error()

