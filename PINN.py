import torch
import torch.nn as nn
from data.dataloaders import DataLoader_Scalar
from datetime import datetime
import pandas as pd
import torch.nn.functional as F
import numpy as np


class PINN2D(nn.Module):
    def __init__(self, layers, parameters, kappa, eta):
        """
        layers: list specifying the number of neurons in each layer
        a, b: advection speeds in x and y directions
        D: diffusion coefficient
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(PINN2D, self).__init__()
        self.params = nn.Parameter(parameters)
        self._kappa = kappa
        self._eta = eta
        
        # Build a fully connected neural network
        self.linears = nn.ModuleList().to(self.device)
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))
        
        # Xavier initialization
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        # Concatenate inputs
        out = torch.cat((x, y, t), dim=1)
        out = torch.tanh(self.linears[0](out))
        # Pass through network
        for i in range(1, len(self.linears) - 1):
            out = out + torch.tanh(self.linears[i](out))
        out = self.linears[-1](out)
        return out
    
    def kappa(self, x, y):
        return self._kappa(x, y, self.params)
    
    def eta(self, x, y):
        return self._eta(x, y, self.params)
    

    def pde_residual(self, x, y, t):
        """
        Physics loss for the 2D advection-diffusion PDE:
        u_t + a*u_x + b*u_y - D*(u_xx + u_yy) = 0
        """
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.forward(x, y, t)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]

        

        kappa = self.kappa(x, y)
        eta = self.eta(x, y)
        
        return u_t + eta*u*(u_x + u_y) - kappa*(u_xx + u_yy)
    
    def initial_condition_loss(self, x, y):
        # Sample points in the spatial domain
        t = torch.zeros_like(x)
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        # Define target as sin(pi*x)*sin(pi*y)
        # (Replace with your actual initial condition)
        u_0 = - torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)) + torch.exp(-((x - 1) ** 2 + (y - 1) ** 2))

        # Model prediction at t=0
        u_pred = self.forward(x, y, t)

        # Mean squared error for the initial condition
        return u_pred - u_0
    
    def boundary_condition_loss(self, x, y, t, L):
        # Model prediction on the boundary
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        u_pred = self.forward(x, y, t)
        # Compute derivatives
        u_x = torch.autograd.grad(u_pred, x, torch.ones_like(u_pred), create_graph=True)[0]
        u_y = torch.autograd.grad(u_pred, y, torch.ones_like(u_pred), create_graph=True)[0]
        
        # Identify top/bottom boundary where y = ±L/2
        # Identify left/right boundary where x = ±L/2
        eps = 1e-7
        is_top_bottom = (torch.abs(torch.abs(y) - (L/2)) < eps)
        is_left_right = (torch.abs(torch.abs(x) - (L/2)) < eps)
        
        # Initialize a zero vector; we only enforce derivative = 0 at boundary
        bc_error = torch.zeros_like(u_pred)
        # Zero normal derivative for top/bottom => u_y = 0
        bc_error[is_top_bottom] = u_y[is_top_bottom]
        # Zero normal derivative for left/right => u_x = 0
        bc_error[is_left_right] = u_x[is_left_right]
        
        # Return the boundary error; for training, use torch.mean(bc_error**2)
        return bc_error

def train_pinn(model, optimizer, L, T, x_train, y_train, t_train, u_train, num_epochs=10000):
    """
    Example training loop using random collocation points.
    For a real application, supply boundary/initial condition
    losses and domain collocation points.
    """
    x = torch.linspace(-3, 3, 50).to(device)
    y = torch.linspace(-3, 3, 50).to(device)
    t = torch.linspace(0, T, 100).to(device)[1:]
    grid_x, grid_y, grid_t = torch.meshgrid(x, y, t, indexing='ij')


    # High definition solution
    data_high = np.load(folder + '/phys.npy')
    data_high = torch.from_numpy(data_high).float().to(device)
    mse = torch.zeros(epoch).to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        # Sample points for initial condition
        x0 = torch.rand(1000, 1).to(device)*L - L/2
        y0 = torch.rand(1000, 1).to(device)*L - L/2
        # Sample points for residual
        x = torch.rand(1000, 1).to(device)*L - L/2
        y = torch.rand(1000, 1).to(device)*L - L/2
        t = torch.rand(1000, 1).to(device)*T
        # Sample points for boundary condition
        x_bc_1 = torch.rand(1000//4, 1).to(device)*L - L/2
        y_bc_1 = torch.full_like(x_bc_1, -L/2).to(device)
        x_bc_2 = torch.rand(1000//4, 1).to(device)*L - L/2
        y_bc_2 = torch.full_like(x_bc_2, L/2).to(device)
        x_bc_3 = torch.full_like(x_bc_1, -L/2).to(device)
        y_bc_3 = torch.rand(1000//4, 1).to(device)*L - L/2
        x_bc_4 = torch.full_like(x_bc_1, L/2).to(device)
        y_bc_4 = torch.rand(1000//4, 1).to(device)*L - L/2
        x_bc = torch.cat((x_bc_1, x_bc_2, x_bc_3, x_bc_4), dim=0)
        y_bc = torch.cat((y_bc_1, y_bc_2, y_bc_3, y_bc_4), dim=0)
        n_bc = x_bc.shape[0]
        t_bc = torch.rand(n_bc, 1).to(device)*T

        
        # IC loss
        ic_loss = model.initial_condition_loss(x0, y0)
        # PDE loss
        res = model.pde_residual(x, y, t)
        # BC loss
        bc_loss = model.boundary_condition_loss(x_bc, y_bc, t_bc, L)

        # Compute loss
        loss_phys = torch.mean(res**2) + 10*torch.mean(ic_loss**2) + torch.mean(bc_loss**2) 

        u_pred = model(x_train, y_train, t_train)
        loss_data = torch.mean((u_pred - u_train)**2)
       
        
        # fix the params until loss_phys < 0.1 
        if loss_phys > 0.1:
            model.params.requires_grad = False
            loss = loss_phys
        else:
             model.params.requires_grad = True
             loss = loss_data + loss_phys

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Calculate the mse on the high definition grid and solution
            data_interp = F.interpolate(data_high.unsqueeze(1), size=(grid_x.shape[0], grid_x.shape[1]), mode='bilinear', align_corners=True)
            data_interp = data_interp.squeeze(1)  # shape (99, 100, 100)
            u_pred = model(grid_x, grid_y, grid_t)
            print(u_pred.shape)
            mse_phys = torch.mean((u_pred - data_interp) ** 2)
            mse[epoch] = mse_phys.item()
            
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, loss_data: {loss_data.item()}, loss_phys: {loss_phys.item()}, mse_phys: {mse_phys.item()}")
    return mse 
import matplotlib.pyplot as plt
import numpy as np

def visualize_pinn_solution(model, L, t_value=0.5, nx=50, ny=50):
    # Create a mesh for x and y
    x_vals = np.linspace(0, 1, nx)*L - L/2
    y_vals = np.linspace(0, 1, ny)*L - L/2
    X, Y = np.meshgrid(x_vals, y_vals)

    # Convert to torch tensors
    x_torch = torch.tensor(X.ravel(), dtype=torch.float32).unsqueeze(1)
    y_torch = torch.tensor(Y.ravel(), dtype=torch.float32).unsqueeze(1)
    t_torch = torch.full_like(x_torch, t_value)
    model = model.to(torch.device("cpu"))
    # Predict
    with torch.no_grad():
        u_pred = model(x_torch, y_torch, t_torch)

    U = u_pred.cpu().numpy().reshape(X.shape)

    # Plot a contour of the solution
    plt.figure(figsize=(6, 5))
    cp = plt.contourf(X, Y, U, levels=100, cmap='viridis')
    plt.colorbar(cp)
    plt.title(f"PINN solution at t={t_value}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"pinn_solution{t_value}.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = "data/adv_diff"
    data = DataLoader_Scalar(device, folder)
    
    x_train, y_train, u_train = data.u[:,1:,0], data.u[:,1:,1], data.u[:,1:,2]
    t_train = torch.zeros_like(x_train.T).to(device)
    for i in range(t_train.shape[0]):
        t_train[:,i] = data.t[1:]

    # Example usage
    def kappa(x, y, parameters):
        a, ax, ay = parameters[0:3]
        return 0.1*torch.ones_like(x) + a*torch.exp(-((x - ax)**2 + (y - ay)**2))
    
    def eta(x, y, parameters):
        b, bx, by = parameters[3:6]
        return 0.1*torch.ones_like(x) + b*torch.exp(-((x - bx)**2 + (y - by)**2))
    

    net_layers = [3, 500, 500, 500, 500, 500, 1]  # (x, y, t) -> u
    parameters = torch.tensor([1, 1.4, -1.3, 1, -2, -1.1]).float().to(device)
    L = 6
    T = 1
    model = PINN2D(net_layers, parameters, kappa=kappa, eta=eta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = train_pinn(model, optimizer, L, T,
                x_train.reshape(-1,1), y_train.reshape(-1,1),
                t_train.reshape(-1,1), u_train.reshape(-1,1), num_epochs=3000)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    parameters = np.concatenate((parameters[0:3].cpu().numpy(), np.array([1.0]), parameters[3:6].cpu().numpy(), np.array([1.0])))
    params = pd.DataFrame(parameters, columns=["params"])
    errors = pd.DataFrame(mse, columns=["mse"])
    params.to_csv(f'parameters/adv_diff/param_pinn_{timestamp}.csv', index=False)
    errors.to_csv(f'parameters/adv_diff/error_pinn_{timestamp}.csv', index=False)


    # Visualize the solution at t=0.5
    for i in np.linspace(0, 1, 11):
        visualize_pinn_solution(model, L, t_value=i, nx=50, ny=50)

