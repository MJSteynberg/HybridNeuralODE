from data.dataloaders import Dissipative
import torch
from torch.utils.data import DataLoader, Subset
from models.training_new import Trainer
from models.neural_odes_new import NeuralODE
from models.plot import plot_results_separate
from models.FEM import HeatEquation2M, AdvectionDiffusionReaction2M
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')
data_dim = 3
hidden_dim = 1000
num_steps = 100
T = 0.2
nX = 150

anode = NeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='relu',
                      architecture='bottleneck', T=T, time_steps=num_steps).to(device)

# Load the state dictionary from the file
state_dict = torch.load("Experiments/2materials/anode_1.pt")

# Load the state dictionary into the model
anode.load_state_dict(state_dict)

#initialize on grid
def create_grid(start, end, step):
    x = torch.arange(start, end + step, step, device=device)
    y = torch.arange(start, end + step, step, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    grid = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), torch.zeros((grid_x.numel(), 1), device=device)],
                     dim=1)
    x, y = grid[:, 0], grid[:, 1]
    
    L = 6
    u0 =(torch.cos(2*torch.pi*x/L) * torch.cos(2*torch.pi*y/L))
    grid[:,2] = u0
    plate_length = int(np.sqrt(grid.shape[0]))
    u0_reshaped = u0.reshape(plate_length, plate_length)
    
    

    return grid, u0_reshaped

grid, u0 = create_grid(-4.5, 4.5, 9/(nX-1))

# predict via node 
pred = anode.f(grid)
fig = plt.figure()
ax = fig.subplots()
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
for i in range(0,num_steps-1, 10):
    ax.scatter(pred[i,:, 0].detach(), pred[i,:, 1].detach() , c=pred[i,:, 2].detach(), cmap='viridis', vmin = torch.min(pred[0,:, 2].flatten()), vmax = torch.max(pred[0,:, 2].flatten()), s = 10)
    plt.savefig(f"Experiments/2materials/fig_{i}_.png")
    ax.cla()
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    


