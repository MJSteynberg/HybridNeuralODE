from data.dataloaders import Dissipative
import torch
import torch.nn as nn
from models.trainingGPT import Trainer
from models.neural_odes import NeuralODE
from models.plot import plot_errors_separate, plot_results_separate
from models.utils import calculate_errors
from models.FEM import heatsolve
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cpu')


def interpolate_heat_solution(grid_traj_forward, heat_solution, plate_length=21, plot_results=False):
    """
    Interpolate the heat solution onto the trajectory points for each time step using bilinear interpolation.

    Args:
    - grid_traj_forward (tensor): A tensor of shape (101, 441, 3) containing the trajectories (x, y, u).
    - heat_solution (tensor): A tensor of shape (101, 21, 21) containing the heat solution over time.

    Returns:
    - interpolated_solution (tensor): A tensor of shape (101, 441) containing the interpolated heat values
                                      for each trajectory at each time step.
    """

    # Assuming grid is [-3, 3] in both x and y for a 21x21 grid
    x_grid = torch.linspace(-3, 3, plate_length, device=grid_traj_forward.device)
    y_grid = torch.linspace(-3, 3, plate_length, device=grid_traj_forward.device)

    interpolated_solution = torch.zeros(grid_traj_forward.shape[0], grid_traj_forward.shape[1],
                                        device=grid_traj_forward.device)

    # Iterate over time steps (101)
    for t in range(grid_traj_forward.shape[0]):
        traj_points = grid_traj_forward[t, :, :2]  # Extract the x and y coordinates

        # Bilinear interpolation
        x_indices = ((traj_points[:, 0] - x_grid[0]) / (x_grid[1] - x_grid[0])).clamp(0, plate_length - 1)
        y_indices = ((traj_points[:, 1] - y_grid[0]) / (y_grid[1] - y_grid[0])).clamp(0, plate_length - 1)

        x0 = x_indices.floor().long()
        x1 = (x0 + 1).clamp(0, plate_length - 1)
        y0 = y_indices.floor().long()
        y1 = (y0 + 1).clamp(0, plate_length - 1)

        # Gather values for interpolation
        Q11 = heat_solution[t, y0, x0]
        Q21 = heat_solution[t, y0, x1]
        Q12 = heat_solution[t, y1, x0]
        Q22 = heat_solution[t, y1, x1]

        # Bilinear interpolation formula
        interpolated_values = (Q11 * (x1 - x_indices) * (y1 - y_indices) +
                               Q21 * (x_indices - x0) * (y1 - y_indices) +
                               Q12 * (x1 - x_indices) * (y_indices - y0) +
                               Q22 * (x_indices - x0) * (y_indices - y0))

        interpolated_solution[t, :] = interpolated_values

        if plot_results:
            # Plotting interpolated solution as trajectories
            plt.figure(figsize=(10, 6))
            for t in range(0, grid_traj_forward.shape[0], 10):  # Plot every 10th time step
                traj_points = grid_traj_forward[t, :, :2].cpu().numpy()
                interpolated_values = interpolated_solution[t, :].cpu().numpy()
                plt.scatter(traj_points[:, 0], traj_points[:, 1], c=interpolated_values, cmap='viridis', s=10,
                            label=f'Time {t}')

            plt.colorbar(label='Interpolated Temperature')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.title('Interpolated Heat Solution Over Trajectories')
            plt.legend()
            plt.show()

    return interpolated_solution


class HeatSolver(nn.Module):
    def __init__(self, device, T=10, nX=21, time_steps=101):
        super(HeatSolver, self).__init__()
        self.device = device
        self.T = T
        self.nX = nX
        self.time_steps = time_steps

        # Make alpha a trainable parameter
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=self.device), requires_grad=True)

    def forward(self, u0, target):
        delta_x = 6 / (self.nX - 1)
        delta_t = self.T / (self.time_steps - 1)
        gamma = (self.alpha * delta_t) / (delta_x ** 2)

        # Initialize solution as a tensor
        u = torch.empty((self.time_steps, self.nX, self.nX), dtype=torch.float, device=self.device)
        u[0, :, :] = u0

        # Boundary conditions (fixed temperature)
        u_top = 0.0
        u_left = 0.0
        u_bottom = 0.0
        u_right = 0.0

        u[:, -1, :] = u_top
        u[:, :, 0] = u_left
        u[:, 0, :] = u_bottom
        u[:, :, -1] = u_right

        for k in range(self.time_steps - 1):
            u[k + 1, 1:-1, 1:-1] = gamma * (
                    u[k, 2:, 1:-1] + u[k, :-2, 1:-1] + u[k, 1:-1, 2:] + u[k, 1:-1, :-2] - 4 * u[k, 1:-1, 1:-1]
            ) + u[k, 1:-1, 1:-1]

        final = interpolate_heat_solution(target, u)

        return final


def train_model(model, observed_data, u0, num_epochs=500, learning_rate=0.1):
    optimizer = torch.optim.Adam([model.alpha], lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        u_pred = model(u0, observed_data)
        loss = criterion(u_pred, observed_data[:, :, 2])
        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, Alpha: {model.alpha.item()}')

    return model.alpha.item()


def create_grid(start, end, step):
    # Generate the x and y coordinates with the specified step size
    x = torch.arange(start, end + step, step)
    y = torch.arange(start, end + step, step)

    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    # Stack and reshape into grid with shape (points x points, 2)
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

    # Add a column of zeros to make it (points^2, 3)
    grid = torch.cat([grid, torch.zeros((grid.shape[0], 1))], dim=1)

    # Define u0 as exp(-5x^2 - 5y^2) on the grid
    x_grid, y_grid = grid[:, 0], grid[:, 1]
    u0 = 5*torch.exp(-5 * ((x_grid+1) ** 2 + (y_grid+1) ** 2)) - \
            3*torch.exp(-5 * ((x_grid-1) ** 2 + (y_grid-1) ** 2))


    # Reshape u0 to 21x21 grid
    plate_length = int(np.sqrt(grid.shape[0]))
    u0_reshaped = u0.reshape(plate_length, plate_length)

    # Return both the grid and u0
    return grid.to(device), u0_reshaped.to(device)


def main():
    # =============================================================================
    # DATA
    # =============================================================================
    # Load from MATLAB
    filename_data = '../u.mat'
    datasets = Dissipative(filename_data)
    t, u = datasets[:]

    # separate the data into u_train and u_test
    train_size = int(0.6 * datasets.length_u())
    shuffled_indices = torch.randperm(datasets.length_u())
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]
    u_train = u[:, train_indices, :].to(device)
    u_test = u[:, test_indices, :].to(device)

    # Define a 441x3 grid of points within [-3, 3] with an added column of zeros
    # Define the initial datum
    grid, u0 = create_grid(-3, 3, 0.3)  # Example for a 21x21 grid

    T, num_steps = max(t).item(), datasets.length_t()
    dt = T / (num_steps - 1)
    # Example Usage:
    model = HeatSolver(device, time_steps=num_steps, T=T)

    # Train to find optimal alpha
    optimal_alpha = train_model(model, u_train, u0)
    print("Optimal alpha:", optimal_alpha)


if __name__ == "__main__":
    main()
