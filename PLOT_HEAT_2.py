
import torch 
from models.FEM import AdvectionDiffusion, Heat
import numpy as np
import matplotlib.pyplot as plt
from data.dataloaders import DataLoader_Scalar
from models.neural_odes import NeuralODE

def interpolate_phys_solution(data, phys_solution, nX=11):
    """
    Interpolate the phys solution onto the trajectory points for
    each time step using bilinear interpolation.

    Args:
    - data (tensor): A tensor of shape (T, N, 3)
                        containing the trajectories (x, y, u).
    - phys_solution (tensor): A tensor of shape (T, nX, nX)
                        containing the phys solution over time.

    Returns:
    - interpolated_solution (tensor): A tensor of shape (T, N)
                        containing the interpolated phys values
                        for each trajectory at each time step.
    """

    # Normalize query points across all time steps
    query_points = data[..., :2]  # Extract (x, y) coordinates
    query_normalized = (query_points + 3) / 3 - 1
    query_normalized = query_normalized.unsqueeze(1)

    # Prepare phys_solution for batch processing
    grid_values = phys_solution.unsqueeze(1)  # Shape: (T, 1, H, W)
    
    # Perform batch grid sampling
    interpolated = nn.functional.grid_sample(
        grid_values, query_normalized, mode='bilinear', align_corners=True
    )  # Output shape: (T, 1, N, 1)
    # Squeeze to match output shape
    interpolated_solution = interpolated.squeeze(1).squeeze(1)  # Shape: (T, N)

    return interpolated_solution


def create_grid(start, end, step):
    device = torch.device('cpu')
    x = torch.arange(start, end + step, step, device=device)
    y = torch.arange(start, end + step, step, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    grid = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), torch.zeros((grid_x.numel(), 1), device=device)], dim=1)
    x, y = grid[:, 0], grid[:, 1]
    u0 = - torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)) + torch.exp(-((x - 1) ** 2 + (y - 1) ** 2))
    plate_length = int(np.sqrt(grid.shape[0]))
    u0_reshaped = u0.reshape(plate_length, plate_length)

    return grid, u0_reshaped

def generate_physical_solution(alpha):
    num_steps = 5000
    L = 6.0  # from -3 to 3
    N = 100
    T = 1
    num_gaussians = 2

    dt = T / num_steps
    device = "cpu"

    # Create the heat equation model
    adv_diff = Heat(device, L, N, dt, num_steps + 1, num_gaussians, alpha = alpha)

    # Create a sum of gaussians initial condition
    x = torch.linspace(-3,3, N)
    y = torch.linspace(-3,3, N)
    x, y = torch.meshgrid(x, y, indexing='ij')
    u0 = torch.zeros(N, N)

    u0 = - torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)) + torch.exp(-((x - 1) ** 2 + (y - 1) ** 2))
    solution = adv_diff(u0)

    return solution[::1000]

def generate_synthetic_solution():
    nX = 100
    hidden_dim = 1000
    device = torch.device('cpu')
    split = 1
    folder = 'data/heat/full'
    data = DataLoader_Scalar(device, folder)
    train_size = int(split*data.length_u())
    indices = torch.randperm(data.length_u())[:train_size]
    device = torch.device('cpu')
    # Obtain variables from data
    L = 6
    u_train = data.u[:, indices, :]
    grid, u0 = create_grid(-L//2, L//2, L/(nX-1))
    data_dim = data.u.shape[2]
    T = data.t[-1]
    num_steps = data.length_t()
    dt = data.t[1] - data.t[0]
    # create models, optimizers and schedulers
    node = NeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='relu', T=T, time_steps=num_steps).to(device)
    #phys = AdvectionDiffusion(device, L, nX, dt, num_steps, num_gaussians_alpha, num_gaussians_kappa, alpha = alpha, kappa = kappa)
    node.load_state_dict(torch.load('models/heat/node.pt'))
    #phys.load_state_dict(torch.load('models/adv_diff/phys.pt'))
    L = 10
    grid, u0 = create_grid(-L//2, L//2, L/(nX-1))
    #u_phys = phys(u0)
    # as input we need (x,y,u) using the grid
    x = grid[:, 0].reshape(-1,1)
    y = grid[:, 1].reshape(-1,1)
    u = u0.reshape(-1,1)
    init = torch.cat([x, y, u], dim=1)
    print(x.shape, y.shape, u.shape, init.shape)
    u_node = node.f(init).detach().numpy()
    # Synthetic solution excludes the initial condition, so add it here.
    u_node = np.concatenate([init.unsqueeze(0), u_node], axis=0)
    return u_node[::20]



# ...existing code...
if __name__ == '__main__':

    alpha_real = torch.tensor([3, 2.5, 1, -2, 1, -2, 1.0, 1.0]).float()
    real_solution = generate_physical_solution(alpha_real)

    model_state_dict = torch.load('models/heat/phys.pt')
    alpha_pred = model_state_dict['alpha'].to('cpu')
    predicted_solution = generate_physical_solution(alpha_pred)

    u_node = generate_synthetic_solution()


    timesteps = [0, 1, 2, 3, 4, -1]
    time_labels = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']

    from matplotlib.gridspec import GridSpec
    # Modified figure size and GridSpec to accommodate 6 rows.
    fig = plt.figure(figsize=(10, 10*12/14))
    gs = GridSpec(
        5, 7,  # Changed from 5 to 6 rows.
        width_ratios=[1, 1, 1, 1, 1, 1, 0.2],
        wspace=0.1, hspace=0.1,
        left=0.05, right=0.93, top=0.93, bottom=0.08
    )

    # Create a 5x6 grid for real/predicted/error and two rows for node solutions.
    axes = []
    for row in range(5):
        row_axes = []
        for col in range(6):
            ax = fig.add_subplot(gs[row, col])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            row_axes.append(ax)
        axes.append(row_axes)

    # Last column is for two colorbars already allocated for real/pred and error.
    cax_sol = fig.add_subplot(gs[0:3, 6])  # colorbar for real/pred
    cax_err = fig.add_subplot(gs[3:5, 6])    # colorbar for error

    # Compute global min/max for real and predicted solutions.
    all_values = real_solution.flatten()
    vmin_sol = all_values.min().item()
    vmax_sol = all_values.max().item()

    all_errors = (real_solution - predicted_solution).abs()
    vmin_err = 0 #all_errors.min().item()
    vmax_err = 0.3 #all_errors.max().item()

    # Top row: Real solution.
    for i, t_idx in enumerate(timesteps):
        ax = axes[0][i]
        im_sol = ax.imshow(
            real_solution[t_idx].detach().cpu().numpy(),
            extent=(-3, 3, -3, 3),
            origin='lower',
            vmin=vmin_sol,
            vmax=vmax_sol
        )
        if i == 0:
            ax.set_ylabel('True')
        ax.set_title(f't = {time_labels[i]}')

    # Middle row: Predicted solution.
    for i, t_idx in enumerate(timesteps):
        ax = axes[1][i]
        im_sol = ax.imshow(
            predicted_solution[t_idx].detach().cpu().numpy(),
            extent=(-3, 3, -3, 3),
            origin='lower',
            vmin=vmin_sol,
            vmax=vmax_sol
        )
        if i == 0:
            ax.set_ylabel('Physical')

    # Third row: Absolute error between real and predicted.
    for i, t_idx in enumerate(timesteps):
        ax = axes[3][i]
        error = all_errors[t_idx].detach().cpu().numpy()
        im_err = ax.imshow(
            error,
            extent=(-3, 3, -3, 3),
            origin='lower',
            vmin=vmin_err,
            vmax=vmax_err
        )
        if i == 0:
            ax.set_ylabel('Physical Error')

    #

    # Fourth row: Scatter plots for node solution.
    for i, t_idx in enumerate(timesteps):
        ax = axes[2][i]
        ax.scatter(u_node[t_idx,:,0], u_node[t_idx,:,1], c=u_node[t_idx,:,2], cmap='viridis', s=5, vmin = vmin_sol, vmax = vmax_sol)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        if i == 0:
            ax.set_ylabel('Synthetic')
    
    # Fifth row: Duplicate scatter plots for node solution.
        
    # --- New code for node solution interpolation and error plotting ---
    import numpy as np
    from scipy.interpolate import griddata

    # Create a grid matching the real solution's resolution.
    ngrid = real_solution.shape[1]  # Assumes square grid (e.g., 30).
    xcoords = np.linspace(-3, 3, ngrid)
    ycoords = np.linspace(-3, 3, ngrid)
    grid_x, grid_y = np.meshgrid(xcoords, ycoords, indexing='ij')

    # Interpolate the node solution onto the grid for each time step.
    node_interp = []
    for t in range(u_node.shape[0]):
        points = u_node[t, :, :2]   # (x, y) coordinates.
        values = u_node[t, :, 2]      # corresponding u values.
        grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
        node_interp.append(grid_z)
    node_interp = np.array(node_interp)

    # Add a new row (row 5) to plot the node error (|real - node_interp|).
    axes_node_err = []
    for i, t_idx in enumerate(timesteps):
        ax = fig.add_subplot(gs[4, i])
        error_node = np.abs(real_solution[t_idx].detach().cpu().numpy() - node_interp[t_idx])
        im_node_err = ax.imshow(
            error_node,
            extent=(-3, 3, -3, 3),
            origin='lower',
            vmin=vmin_err,
            vmax=vmax_err
        )
        if i == 0:
            ax.set_ylabel('Synthetic Error')
        ax.set_xticks([])
        ax.set_yticks([])
        axes_node_err.append(ax)


    # --- End new code ---

    # Colorbars for real/pred and predicted error.
    fig.colorbar(im_sol, cax=cax_sol, orientation='vertical')
    fig.colorbar(im_err, cax=cax_err, orientation='vertical')

    plt.savefig('comparison_heat.png', dpi=300)
# ...existing code...