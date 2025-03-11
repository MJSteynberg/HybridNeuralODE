import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from models.FEM import AdvectionDiffusion
from models.training import interpolate_phys_solution
import numpy as np

# Define the ODE system
class SimpleHarmonicOscillator(torch.nn.Module):
    def forward(self, t, state):
        matrix = torch.tensor([[-0.1, -1], [1, -0.1]], dtype=torch.float32)
        return matrix @ (state)

# Solve the ODE
def solve_ode_with_odeint(x0, y0, t):
    """
    Solves the ODE system x' = y, y' = -x using torchdiffeq's odeint.
    
    Args:
    - x0, y0 (float): Initial conditions for x and y.
    - t (torch.Tensor): Time points for evaluation.
    
    Returns:
    - torch.Tensor: Solution (len(t), 2) where [:, 0] is x(t) and [:, 1] is y(t).
    """
    # Initial conditions as a torch.Tensor
    initial_state = torch.tensor([x0, y0], dtype=torch.float32)
    
    # Instantiate the ODE function
    ode_func = SimpleHarmonicOscillator()
    
    # Solve the ODE
    solution = odeint(ode_func, initial_state, t)
    return solution
    
def advectiondiffusion(flag = "partial"):
    
    num_steps = 10000
    L = 6.0  # from -3 to 3
    N = 100
    T = 1
    num_gaussians_alpha = 1
    num_gaussians_kappa = 1
    alpha = torch.tensor([3,  1, 1, 1.0]).float() # [Amplitude, Amplitude, x0, x0, y0, y0, sigma, sigma]
    kappa = torch.tensor([2.5, -2, -2, 1.0]).float() # [Amplitude, Amplitude, x0, x0, y0, y0, sigma, sigma]
    

    dt = T / num_steps
    sqrt_num_sensors = 10 # 
    device = "cpu"

    # Create the heat equation model
    adv_diff = AdvectionDiffusion(device, L, N, dt, num_steps + 1, num_gaussians_alpha, num_gaussians_kappa, alpha = alpha, kappa = kappa)

    # Create a sum of gaussians initial condition
    x = torch.linspace(-3,3, N)
    y = torch.linspace(-3,3, N)
    x, y = torch.meshgrid(x, y, indexing='ij')
    u0 = torch.zeros(N, N)

    u0 = - torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)) + torch.exp(-((x - 1) ** 2 + (y - 1) ** 2))
    heat_solution = adv_diff(u0)

    if flag == "partial":
        # Create sens   or initial conditions in [1,2]
        a = torch.linspace(1, 2, sqrt_num_sensors)
        b = torch.linspace(1, 2, sqrt_num_sensors)
        x0, y0 = torch.meshgrid(a, b, indexing='ij')
        x0 = x0.flatten()
        y0 = y0.flatten()
    else:
        # Create sensors in [-3,3]
        a = torch.linspace(-3, 3, sqrt_num_sensors)
        b = torch.linspace(-3, 3, sqrt_num_sensors)
        x0, y0 = torch.meshgrid(a, b, indexing='ij')
        # if x0^2 + y0^2 > 3^2 remove the point
        mask = x0**2 + y0**2 <= 3**2
        x0 = x0[mask]
        y0 = y0[mask]
        x0 = x0.flatten()
        y0 = y0.flatten()

    data_pts = x0.shape[0]

    data = torch.empty((num_steps, data_pts, 3))
    t = torch.linspace(0, T, num_steps)
    for i in range(data_pts):
        data[:,i,:2]= solve_ode_with_odeint(x0[i], y0[i], t)

    # plot the data
    for i in range(data_pts):
        plt.scatter(data[:,i,0].detach(), data[:,i,1].detach(), c = 'k')
    plt.savefig("sensors.png")

    data[:,:,2] = interpolate_phys_solution(data, heat_solution)
    
    # plot the data
    for i in range(data_pts):
        plt.plot(data[:,i,0].detach(), data[:,i,1].detach())
    plt.show()

    #extract every fifth data point
    data = data[::100,:,:]
    # save the data 
    import numpy as np
    if flag == "partial":
        np.save('data/adv_diff/data.npy', data.detach().numpy())
    else:
        np.save(f'data/adv_diff/{flag}/data.npy', data.detach().numpy())
    
    
