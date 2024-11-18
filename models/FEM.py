import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


def plot_heatmap(u, title="Heat Distribution"):
    plt.figure(figsize=(6, 5))
    plt.imshow(u.cpu().numpy(), cmap='hot', origin='lower')
    plt.colorbar(label="Temperature")
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()


class heatsolve(nn.Module):
    def __init__(self,
                 device,
                 T=10,
                 nX=21,
                 time_steps=10,
                 param_grid = 1):
        super(heatsolve, self).__init__()
        self.device = device
        self.T = T
        self.nX = nX
        self.time_steps = time_steps
        self.param_grid = param_grid
        alpha = torch.ones((param_grid, param_grid), device=device)
        

        # Make alpha a trainable parameter
        self.alpha = nn.Parameter(alpha, requires_grad=True)
        
        self.delta_x = 6 / (self.nX - 1)

        # Calculated params
        self.delta_t = self.T / (self.time_steps - 1)
        

    def forward(self, u0):
        # Initialize solution: the grid of u(k, i, j) as a tensor
        u = torch.empty((self.time_steps, self.nX, self.nX), dtype=torch.float, device=self.device)
        gamma = (F.interpolate(self.alpha.unsqueeze(0).unsqueeze(0), size=(self.nX, self.nX), mode='nearest') * self.delta_t / (self.delta_x ** 2)).squeeze(0).squeeze(0)
        # Boundary conditions (fixed temperature)
        u_top = 0.0
        u_left = 0.0
        u_bottom = 0.0
        u_right = 0.0

        # Set the initial condition
        u[0, :, :] = u0

        # Set the boundary conditions
        u[:, (self.nX - 1):, :] = u_top
        u[:, :, :1] = u_left
        u[:, :1, 1:] = u_bottom
        u[:, :, (self.nX - 1):] = u_right

        
        for k in range(self.time_steps - 1):
            u[k + 1, 1:-1, 1:-1] = gamma[1:-1, 1:-1] * (
                    u[k, 2:, 1:-1] + u[k, :-2, 1:-1] + u[k, 1:-1, 2:] + u[k, 1:-1, :-2] - 4 * u[k, 1:-1, 1:-1]
            ) + u[k, 1:-1, 1:-1]

        return u


class wavesolve(nn.Module):
    def __init__(self,
                 device,
                 T=10,
                 nX=21,
                 time_steps=10):
        super(wavesolve, self).__init__()
        self.device = device
        self.T = T
        self.nX = nX
        self.time_steps = time_steps

        # Make alpha a trainable parameter
        self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32, device=self.device), requires_grad=True)
        
        self.delta_x = 6 / (self.nX - 1)

        # Calculated params
        self.delta_t = self.T / (self.time_steps - 1)
        

    def forward(self, u0):
        # Initialize solution: the grid of u(k, i, j) as a tensor
        u = torch.empty((self.time_steps, self.nX, self.nX), dtype=torch.float, device=self.device)
        
        gamma = (self.alpha ** 2 * self.delta_t ** 2) / (self.delta_x ** 2)
        # Boundary conditions (fixed temperature)
        u_top = 0.0
        u_left = 0.0
        u_bottom = 0.0
        u_right = 0.0

        u[0, :, :] = u0
        u[1, :, :] = u0 + gamma / 2 * (u[0, 2:, 1:-1] + u[0, :-2, 1:-1] + u[0, 1:-1, 2:] + u[0, 1:-1, :-2] - 4 * u[0, 1:-1, 1:-1])

        # Set the boundary conditions
        u[:, (self.nX - 1):, :] = u_top
        u[:, :, :1] = u_left
        u[:, :1, 1:] = u_bottom
        u[:, :, (self.nX - 1):] = u_right

        for k in range(1, self.time_steps - 1):
            u[k + 1, 1:-1, 1:-1] = gamma * (
                    u[k, 2:, 1:-1] + u[k, :-2, 1:-1] + u[k, 1:-1, 2:] + u[k, 1:-1, :-2] - 4 * u[k, 1:-1, 1:-1]
            ) + 2*u[k, 1:-1, 1:-1] - u[k-1, 1:-1, 1:-1]
            
        return u
