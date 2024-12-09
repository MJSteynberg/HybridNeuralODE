import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from FEM import AdvectionDiffusion, HeatEquation, AdvectionDiffusionG, HeatEquationG, WaveEquationG
from training_new import interpolate_heat_solution
import numpy as np

# Define the ODE system
class SimpleHarmonicOscillator(torch.nn.Module):
    def forward(self, t, state):
        matrix = torch.tensor([[-0.1, -1], [1, -0.1]], dtype=torch.float32)
        return matrix @ (state - 1.5)

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

def advectiondiffusion():
    num_steps = 3000
    L = 6.0  # from -3 to 3
    N = 50
    T = 0.2
    param_x = 1
    param_y = 1
    k = torch.tensor([0.8]).float().reshape(1,1)
    alpha = torch.tensor([1]).float().reshape(1,1)
    

    dt = T / num_steps
    sqrt_num_sensors = 10 # 
    device = "cpu"

    # Create the heat equation model
    heat = AdvectionDiffusion(device, L, N, dt, num_steps + 1, param_x, param_y, alpha, k)

    # Create a sum of gaussians initial condition
    x = torch.linspace(-3, 3, N)
    y = torch.linspace(-3, 3, N)
    x, y = torch.meshgrid(x, y, indexing='ij')
    u0 = torch.zeros(N, N)

    u0 = torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)/0.8)
    heat_solution = heat(u0)

    # show u0 
    for i in range(0, num_steps, 30):
        plt.imshow(heat_solution[i].detach().reshape(N, N), origin='lower', vmin=0, vmax=1)
        plt.colorbar()
        plt.pause(0.1)
        plt.clf()




    # Create sensor initial conditions in [-2,2]
    a = torch.linspace(-1, 1.3, sqrt_num_sensors)
    b = torch.linspace(-1, 1.3, sqrt_num_sensors)
    x0, y0 = torch.meshgrid(a, b, indexing='ij')
    x0 = x0.flatten()
    y0 = y0.flatten()



    data = torch.empty((num_steps, sqrt_num_sensors**2, 3))
    t = torch.linspace(0, T, num_steps)
    for i in range(sqrt_num_sensors**2):
        data[:,i,:2]= solve_ode_with_odeint(x0[i], y0[i], t)

    data[:,:,2] = interpolate_heat_solution(data, heat_solution)

    #extract every fifth data point
    data = data[::30,:,:]
    # save the data 
    import numpy as np
    np.save('Experiments/2materials/data.npy', data.detach().numpy())
    
def advectiondiffusionG():
    num_steps = 10000
    L = 6.0  # from -3 to 3
    N = 200
    T = 1
    num_gaussians_alpha = 1
    num_gaussians_kappa = 1
    alpha = torch.tensor([1,  1, 1, 1.0]).float() # [Amplitude, Amplitude, x0, x0, y0, y0, sigma, sigma]
    kappa = torch.tensor([1, -2, -1, 1.0]).float() # [Amplitude, Amplitude, x0, x0, y0, y0, sigma, sigma]
    

    dt = T / num_steps
    sqrt_num_sensors = 10 # 
    device = "cpu"

    # Create the heat equation model
    heat = AdvectionDiffusionG(device, L, N, dt, num_steps + 1, num_gaussians_alpha, num_gaussians_kappa, alpha = alpha, kappa = kappa)
    
    # plot the diffusivity map 
    diff_map = heat.create_diffusion_map()
    plt.imshow(diff_map.detach().reshape(N, N), origin='lower', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Diffusivity map')
    plt.show()
    
    advec_map = heat.create_advection_map()
    plt.imshow(advec_map.detach().reshape(N, N), origin='lower', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Advectivity map')
    plt.show()

    # Create a sum of gaussians initial condition
    x = torch.linspace(-3,3, N)
    y = torch.linspace(-3,3, N)
    x, y = torch.meshgrid(x, y, indexing='ij')
    u0 = torch.zeros(N, N)

    u0 = torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)) + torch.exp(-((x - 1) ** 2 + (y - 1) ** 2))
    heat_solution = heat(u0)

    # show u0 
    # for i in range(0, num_steps, 500):
    #     plt.imshow(heat_solution[i].detach().reshape(N, N), origin='lower', vmin=0, vmax=1)
    #     plt.colorbar()
    #     plt.pause(0.1)
    #     plt.clf()




    # Create sensor initial conditions in [1,2]
    a = torch.linspace(1, 2, sqrt_num_sensors)
    b = torch.linspace(1, 2, sqrt_num_sensors)
    x0, y0 = torch.meshgrid(a, b, indexing='ij')
    x0 = x0.flatten()
    y0 = y0.flatten()



    data = torch.empty((num_steps, sqrt_num_sensors**2, 3))
    t = torch.linspace(0, T, num_steps)
    for i in range(sqrt_num_sensors**2):
        data[:,i,:2]= solve_ode_with_odeint(x0[i], y0[i], t)

    data[:,:,2] = interpolate_heat_solution(data, heat_solution)
    
    # plot the data
    for i in range(sqrt_num_sensors**2):
        plt.plot(data[:,i,0].detach(), data[:,i,1].detach())
    plt.show()

    #extract every fifth data point
    data = data[::100,:,:]
    # save the data 
    import numpy as np
    np.save('Experiments/gaussians/data.npy', data.detach().numpy())
    
def waveequationG():
    num_steps = 8000
    L = 6.0  # from -3 to 3
    N = 200
    T = 0.4
    num_gaussians_alpha = 1
    alpha = torch.tensor([3.0, 0.4, 0.6, 2.0]).float() # [Amplitude, Amplitude, x0, x0, y0, y0, sigma, sigma]
    

    dt = T / num_steps
    sqrt_num_sensors = 10 # 
    device = "cpu"

    # Create the heat equation model
    heat = WaveEquationG(device, L, N, dt, num_steps + 1, num_gaussians_alpha, alpha = alpha)
    
    # Create a sum of gaussians initial condition
    x = torch.linspace(-3, 3, N)
    y = torch.linspace(-3, 3, N)
    x, y = torch.meshgrid(x, y, indexing='ij')
    u0 = torch.zeros(N, N)

    u0 = torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)/0.8) + torch.exp(-((x - 1) ** 2 + (y - 1) ** 2)/0.8)
    heat_solution = heat(u0)

    # show u0 
    # for i in range(0, num_steps, 80):
    #     plt.imshow(heat_solution[i].detach().reshape(N, N), origin='lower', vmin=0, vmax=1)
    #     plt.colorbar()
    #     plt.pause(0.1)
    #     plt.clf()




    # Create sensor initial conditions in [-2,2]
    a = torch.linspace(-1, 1.3, sqrt_num_sensors)
    b = torch.linspace(-1, 1.3, sqrt_num_sensors)
    x0, y0 = torch.meshgrid(a, b, indexing='ij')
    x0 = x0.flatten()
    y0 = y0.flatten()



    data = torch.empty((num_steps, sqrt_num_sensors**2, 3))
    t = torch.linspace(0, T, num_steps)
    for i in range(sqrt_num_sensors**2):
        print(i)
        data[:,i,:2]= solve_ode_with_odeint(x0[i], y0[i], t)

    data[:,:,2] = interpolate_heat_solution(data, heat_solution)
    
    # plot the data
    for i in range(sqrt_num_sensors**2):
        plt.plot(data[:,i,0].detach(), data[:,i,1].detach())
    plt.show()

    #extract every fifth data point
    data = data[::80,:,:]
    # save the data 
    import numpy as np
    np.save('Experiments/wave_gaussians/data.npy', data.detach().numpy())
    
def heat():
    num_steps = 100
    L = 6.0  # from -3 to 3
    N = 60
    T = 0.2
    param_x = 5
    param_y = 5
    
    # interpolate the diffusivity to grid 
    
    # Step 1: Create a reference grid (high resolution)
    x_ref = torch.linspace(0,1, 100)
    y_ref = torch.linspace(0,1, 100)
    x_ref_grid, y_ref_grid = torch.meshgrid(x_ref, y_ref, indexing="ij")

    # Step 2: Compute the function values on the reference grid
    z_ref = 0.5*(x_ref_grid**2 + y_ref_grid**2) + 0.1

    
    z_ref = z_ref.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Step 3: Create target grid for interpolation
    x_target = torch.linspace(0,1, param_x)
    y_target = torch.linspace(0,1, param_y)
    x_target_grid, y_target_grid = torch.meshgrid(x_target, y_target, indexing="ij")

    # Normalize the target grid to [-1, 1] for grid_sample
    grid = torch.stack((y_target_grid, x_target_grid), dim=-1)  # Shape: (param_x, param_y, 2)
    grid = grid.unsqueeze(0)  # Add batch dimension

    # Step 4: Interpolate using grid_sample
    alpha = torch.nn.functional.grid_sample(z_ref, grid, mode='bilinear').squeeze()
  

    dt = T / num_steps
    sqrt_num_sensors = 5 # 
    device = "cpu"

    # Create the heat equation model
    heat = HeatEquation(device, L, N, dt, num_steps + 1, param_x, param_y, alpha)

    # Create a sum of gaussians initial condition
    x = torch.linspace(-3, 3, N)
    y = torch.linspace(-3, 3, N)
    x, y = torch.meshgrid(x, y, indexing='ij')
    u0 = torch.zeros(N, N)

    u0 = torch.exp(-((x + 1) ** 2 + (y + 1) ** 2)/0.8)
    heat_solution = heat(u0)

    # show u0 
    # for i in range(num_steps):
    #     plt.imshow(heat_solution[i].detach().reshape(N, N), origin='lower', vmin=0, vmax=1)
    #     plt.colorbar()
    #     plt.pause(0.1)
    #     plt.clf()




    # Create sensor initial conditions in [-2,2]
    a = torch.linspace(-1, 1.3, sqrt_num_sensors)
    b = torch.linspace(-1, 1.3, sqrt_num_sensors)
    x0, y0 = torch.meshgrid(a, b, indexing='ij')
    x0 = x0.flatten()
    y0 = y0.flatten()



    data = torch.empty((num_steps, sqrt_num_sensors**2, 3))
    t = torch.linspace(0, T, num_steps)
    for i in range(sqrt_num_sensors**2):
        data[:,i,:2]= solve_ode_with_odeint(x0[i], y0[i], t)
        
    #plot the data
    for i in range(sqrt_num_sensors**2):
        plt.plot(data[:,i,0], data[:,i,1])
    plt.show()

    data[:,:,2] = interpolate_heat_solution(data, heat_solution)

    print(data.shape)
    # save the data 
    import numpy as np
    np.save('Experiments/smooth/data.npy', data.detach().numpy())
    
    
advectiondiffusionG()