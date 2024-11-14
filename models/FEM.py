import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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
                 time_steps=10):
        super(heatsolve, self).__init__()
        self.device = device
        self.T = T
        self.nX = nX
        self.time_steps = time_steps

        # Make alpha a trainable parameter
        self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32, device=self.device), requires_grad=True)

    def forward(self, u0):
        delta_x = 6 / (self.nX - 1)

        # Calculated params
        delta_t = self.T / (self.time_steps - 1)
        gamma = (self.alpha * delta_t) / (delta_x ** 2)

        # Initialize solution: the grid of u(k, i, j) as a tensor
        u = torch.empty((self.time_steps, self.nX, self.nX), dtype=torch.float, device=self.device)

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
            u[k + 1, 1:-1, 1:-1] = gamma * (
                    u[k, 2:, 1:-1] + u[k, :-2, 1:-1] + u[k, 1:-1, 2:] + u[k, 1:-1, :-2] - 4 * u[k, 1:-1, 1:-1]
            ) + u[k, 1:-1, 1:-1]

        return u
