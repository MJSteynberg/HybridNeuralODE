import torch
import torch.nn as nn
import time


import torch

def interpolate_heat_solution(data, heat_solution, nX=11):
    """
    Interpolate the heat solution onto the trajectory points for
    each time step using bilinear interpolation.

    Args:
    - data (tensor): A tensor of shape (101, N, 3)
                        containing the trajectories (x, y, u).
    - heat_solution (tensor): A tensor of shape (101, 21, 21)
                        containing the heat solution over time.

    Returns:
    - interpolated_solution (tensor): A tensor of shape (101, N)
                        containing the interpolated heat values
                        for each trajectory at each time step.
    """

    # Assuming grid is [-3, 3] in both x and y
    x_grid = torch.linspace(-3, 3, nX, device=data.device)
    y_grid = torch.linspace(-3, 3, nX, device=data.device)

    interpolated_solution = torch.zeros(data.shape[0], data.shape[1],
                                        device=data.device)

    # Iterate over time steps (101)
    for t in range(data.shape[0]):
        traj_points = data[t, :, :2]  # Extract the x and y coordinates

        # Bilinear interpolation
        x_indices = ((data[t, :, 0] - x_grid[0]) / (x_grid[1] - x_grid[0])).clamp(0, nX - 1)
        y_indices = ((data[t, :, 1] - y_grid[0]) / (y_grid[1] - y_grid[0])).clamp(0, nX - 1)

        x0 = x_indices.floor().long()
        x1 = (x0 + 1).clamp(0, nX - 1)
        y0 = y_indices.floor().long()
        y1 = (y0 + 1).clamp(0, nX - 1)

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

    return interpolated_solution








class Trainer:
    def __init__(self,
                 model_node,
                 model_heat,
                 optimizer_node,
                 optimizer_heat,
                 scheduler_anode,
                 scheduler_heat,
                 device,
                 grid,
                 reg='l1',
                 print_freq=50):
        self.model_node = model_node.to(device)
        self.model_heat = model_heat.to(device)
        self.optimizer_node = optimizer_node
        self.optimizer_heat = optimizer_heat
        self.scheduler_anode = scheduler_anode
        self.scheduler_heat = scheduler_heat
        self.reg = reg
        self.device = device
        self.loss_func = nn.MSELoss()
        self.print_freq = print_freq
        self.grid = grid

    def train(self, u, u0, num_epochs):

        u_train = (u[0, :, :].to(self.device))
        u_target = u[1:, :, :].to(self.device)
        
        alpha_list = torch.empty(num_epochs, self.model_heat.nX, self.model_heat.nX)
        loss_list = torch.empty(num_epochs)

        self.num_epochs = num_epochs
        self.start_time = time.time()

        # Training iteration
        for epoch in range(num_epochs):
            loss, loss_FD = self._train_epoch(u_train, u_target, u0)
            alpha_list[epoch] = nn.functional.interpolate(self.model_heat.alpha.view(1, 1, self.model_heat.param_grid, self.model_heat.param_grid), (self.model_heat.nX, self.model_heat.nX), mode='nearest').view(self.model_heat.nX, self.model_heat.nX)
            loss_list[epoch] = loss_FD
            if (epoch + 1) % self.print_freq == 1:
                elapsed_time = (time.time() - self.start_time) / 60.
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs} | Loss: {loss:.3e} | Time: {elapsed_time:.1f} min | "
                    f"Alpha: {torch.max(self.model_heat.alpha):.10f}")

        print(f"Total time: {time.time() - self.start_time:.1f} sec")
        return alpha_list, loss_list

    def _train_epoch(self, u_train, u_target, u0):
        # Compute heat solution using the current value of alpha
        heat_solution = self.model_heat(u0)

        # Compute trajectory predicted by NODE
        #u_pred, traj = self.model_node(u_train)

        # Forward integration for grid
        # grid_pred, grid_traj_forward = self.model_node(self.grid)

        # Interpolate heat at trajectories
        # interpolated_heat_traj = interpolate_heat_solution(grid_traj_forward, heat_solution)
        interpolated_heat_target = interpolate_heat_solution(u_target, heat_solution)

        # # Regularization parameter and calculation
        # lambda_reg = 1e-4 if self.reg == 'l1' else 1e-3
        # ls_reg = 0.0

        # if self.reg == 'l1':
        #     # Use list comprehension to avoid redundant attribute calls
        #     weights = [self.model_node.dynamics.fc1_time_1.weight,
        #                self.model_node.dynamics.fc1_time_2.weight,
        #                self.model_node.dynamics.fc1_time_3.weight,
        #                self.model_node.dynamics.fc3_time_1.weight,
        #                self.model_node.dynamics.fc3_time_2.weight,
        #                self.model_node.dynamics.fc3_time_3.weight]
        #     ls_reg = sum(w.abs().sum() for w in weights)

        # elif self.reg == 'barron':
        #     # Optimized Barron regularization calculation
        #     weights_fc1 = torch.cat([self.model_node.dynamics.fc1_time_1.weight,
        #                              self.model_node.dynamics.fc1_time_2.weight], dim=0)
        #     weights_fc3 = torch.cat([self.model_node.dynamics.fc3_time_1.weight,
        #                              self.model_node.dynamics.fc3_time_2.weight,
        #                              self.model_node.dynamics.fc3_time_3.weight], dim=0)
        #     ls_reg = (weights_fc1.pow(2).sum(dim=1).sqrt() *
        #               weights_fc3.pow(2).sum(dim=1).sqrt().sum())

        # Calculate the loss
        loss_FD = self.loss_func(interpolated_heat_target, u_target[:, :, 2])
        loss = 1000 * loss_FD
                #+ self.loss_func(interpolated_heat_traj, grid_traj_forward[:, :, 2]))

        # Optimizer steps
        #self.optimizer_node.zero_grad()
        self.optimizer_heat.zero_grad()
        loss.backward()
        #self.optimizer_node.step()
        self.optimizer_heat.step()
        self.scheduler_heat.step()

        return loss.item(), loss_FD.item()
