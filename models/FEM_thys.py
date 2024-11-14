import torch
import torch.nn as nn
import time
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


class heatsolve(nn.Module):
    def __init__(self,
                 device,
                 T=1,
                 nX=21,
                 time_steps=100):

        super(heatsolve, self).__init__()
        self.device = device
        self.T = T
        self.nX = nX
        # Make alpha a trainable parameter
        self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32, device=self.device), requires_grad=True)
        
        self.time_steps = time_steps
        self.delta_x = 6 / (self.nX - 1)
        self.delta_t = self.T / (self.time_steps-1)
        self.gamma = (self.alpha * self.delta_t) / (self.delta_x ** 2)

        
        
        print(f"CFL condition: {self.gamma} < 0.25")
        

    def forward(self, u0):

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
            u[k + 1, 1:-1, 1:-1] = self.gamma * (
                    u[k, 2:, 1:-1] + u[k, :-2, 1:-1] + u[k, 1:-1, 2:] + u[k, 1:-1, :-2] - 4 * u[k, 1:-1, 1:-1]
            ) + u[k, 1:-1, 1:-1]
            
        return u


    def plot_heatmap(self, u, title="Heat Distribution"):
        if len(u.shape) == 2:
            plt.figure(figsize=(6, 5))
            plt.imshow(u.detach().cpu().numpy(), cmap='hot', origin='lower')
            plt.colorbar(label="Temperature")
            plt.title(title)
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.show()
        else:
            plt.figure(figsize=(6, 5))
            plt.title(title)
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.imshow(u[0].detach().cpu().numpy(), cmap='hot', origin='lower')
            plt.colorbar(label="Temperature")
            for i in range(1, u.shape[0]):
                plt.imshow(u[i].detach().cpu().numpy(), cmap='hot', origin='lower')
                plt.pause(0.05)
                

class wavesolve(nn.Module):
    def __init__(self,
                 device,
                 T=1,
                 nX=21,
                 time_steps=100):

        super(heatsolve, self).__init__()
        self.device = device
        self.T = T
        self.nX = nX
        # Make alpha a trainable parameter
        self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32, device=self.device), requires_grad=True)
        
        self.time_steps = time_steps
        self.delta_x = 6 / (self.nX - 1)
        self.delta_t = self.T / (self.time_steps-1)
        self.gamma = (self.alpha ** 2 * self.delta_t ** 2) / (self.delta_x ** 2)

        
        
        print(f"CFL condition: {self.gamma} < 0.25")
        

    def forward(self, u0):

        # Initialize solution: the grid of u(k, i, j) as a tensor
        u = torch.empty((self.time_steps, self.nX, self.nX), dtype=torch.float, device=self.device)

        # Boundary conditions (fixed temperature)
        u_top = 0.0
        u_left = 0.0
        u_bottom = 0.0
        u_right = 0.0

         # Set the initial condition
        u[0, :, :] = u0
        u[1, :, :] = u0 + self.gamma / 2 * (u[0, 2:, 1:-1] + u[0, :-2, 1:-1] + u[0, 1:-1, 2:] + u[0, 1:-1, :-2] - 4 * u[0, 1:-1, 1:-1])

        # Set the boundary conditions
        u[:, (self.nX - 1):, :] = u_top
        u[:, :, :1] = u_left
        u[:, :1, 1:] = u_bottom
        u[:, :, (self.nX - 1):] = u_right

        for k in range(1, self.time_steps - 1):
            u[k + 1, 1:-1, 1:-1] = self.gamma * (
                    u[k, 2:, 1:-1] + u[k, :-2, 1:-1] + u[k, 1:-1, 2:] + u[k, 1:-1, :-2] - 4 * u[k, 1:-1, 1:-1]
            ) + 2*u[k, 1:-1, 1:-1] - u[k-1, 1:-1, 1:-1]


    def plot_heatmap(self, u, title="Wave Amplitude"):
        if len(u.shape) == 2:
            plt.figure(figsize=(6, 5))
            plt.imshow(u.detach().cpu().numpy(), cmap='hot', origin='lower')
            plt.colorbar(label="Wave Amplitude")
            plt.title(title)
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.show()
        else:
            plt.figure(figsize=(6, 5))
            plt.title(title)
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.imshow(u[0].detach().cpu().numpy(), cmap='hot', origin='lower')
            plt.colorbar(label="Wave Amplitude")
            for i in range(1, u.shape[0]):
                plt.imshow(u[i].detach().cpu().numpy(), cmap='hot', origin='lower')
                plt.pause(0.05)
        
        
if __name__ == "__main__":
    print("Testing Heat Solver")    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nX = 31
    model = heatsolve(device, T=1, nX=nX, time_steps=200)
    x_grid, y_grid = torch.meshgrid([torch.linspace(-3, 3, nX), torch.linspace(-3, 3, nX)], indexing='ij')
    u0 = 5*torch.exp(-5 * ((x_grid+1) ** 2 + (y_grid+1) ** 2)) - \
            3*torch.exp(-5 * ((x_grid-1) ** 2 + (y_grid-1) ** 2))
            
    u = model(u0)
    model.plot_heatmap(u, title="Heat Distribution")
    
    print("Testing Wave Solver")
    model = wavesolve(device, T=1, nX=nX, time_steps=200)
    u = model(u0)
    model.plot_heatmap(u, title="Wave Amplitude")
    
    
    
    