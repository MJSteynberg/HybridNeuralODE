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
    
################################################################################################
################# For 1 homogeneous material ########################
################################################################################################

class HeatEquation(nn.Module):
    def __init__(self, device, L, N, dt, num_steps, param_x, param_y, alpha = None):
        super(HeatEquation, self).__init__()
        self.device = device
        self.L = L
        self.N = N
        
        self.dx = L / (N - 1)
        self.dt = dt
        self.num_steps = num_steps   
        
        # Placeholders
        self.nX = N
        self.param_x = param_x 
        self.param_y = param_y
        
        
        # Parameters with gradients
        
        if alpha is not None:
            self.alpha = nn.Parameter(alpha)
            if self.alpha.shape != (param_x, param_y):
                raise ValueError("Shape of alpha does not match the parameters.")
        else:
            self.alpha = nn.Parameter(torch.ones((param_x, param_y), dtype=torch.float32, device=self.device))

        
        max_alpha = torch.max(self.alpha)
        if dt > (self.dx ** 2) / (4 * max_alpha):
            raise ValueError("CFL condition violated. Decrease time step or increase spatial resolution.")
        
        self.diffusion_map = self.create_diffusion_map()

    def create_diffusion_map(self):
        # interpolate alpha to the grid
        diffusion_map = torch.nn.functional.interpolate(self.alpha.view(1, 1, self.param_x, self.param_y), size=(self.N, self.N), mode='bilinear', align_corners=True)
        return diffusion_map.squeeze()
    
    def penalization(self):
        return sum([torch.sum(torch.relu(-p)) for p in self.parameters()])

    def step(self, T):
        dx2 = self.dx ** 2

        # Compute updates for interior points
        T_xx = (T[:-2, 1:-1] - 2 * T[1:-1, 1:-1] + T[2:, 1:-1]) / dx2
        T_yy = (T[1:-1, :-2] - 2 * T[1:-1, 1:-1] + T[1:-1, 2:]) / dx2
        alpha = self.diffusion_map[1:-1, 1:-1]

        # Compute the new temperature
        T_new = T.clone()
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + self.dt * alpha * (T_xx + T_yy)

        # Apply Dirichlet boundary conditions
        T_new[0, :] = T_new[-1, :] = T_new[:, 0] = T_new[:, -1] = 0.0

        return T_new

    def forward(self, u0):
        """
        Simulate the heat equation over a number of time steps.
        """
        self.diffusion_map = self.create_diffusion_map()
        T = u0.clone().requires_grad_(True).to(self.device)  # Start with the initial condition
        intermediate_steps = torch.zeros(self.num_steps, self.N, self.N, device=self.device)

        for t in range(self.num_steps):
            intermediate_steps[t] = T.clone()  # Store the current state
            T = self.step(T)  # Compute the next state
        return intermediate_steps[1:]
    
class AdvectionDiffusion(nn.Module):
    def __init__(self, device, L, N, dt, num_steps, param_x, param_y, alpha = None, kappa = None):
        super(AdvectionDiffusion, self).__init__()
        self.device = device
        self.L = L
        self.N = N
        
        self.dx = L / (N - 1)
        self.dt = dt
        self.num_steps = num_steps   
        
        # Placeholders
        self.nX = N
        self.param_x = param_x 
        self.param_y = param_y
        
        
        # Parameters with gradients
        
        if alpha is not None:
            self.alpha = nn.Parameter(alpha).to(self.device)
            if self.alpha.shape != (param_x, param_y):
                raise ValueError("Shape of alpha does not match the parameters.")
        else:
            self.alpha = nn.Parameter(torch.ones((param_x, param_y), dtype=torch.float32, device=self.device))
            
        if kappa is not None:
            self.kappa = nn.Parameter(kappa).to(self.device)
            if self.kappa.shape != (param_x, param_y):
                raise ValueError("Shape of alpha does not match the parameters.")
        else:
            self.kappa = nn.Parameter(torch.ones((param_x, param_y), dtype=torch.float32, device=self.device))

        
        max_alpha = torch.max(self.alpha)
        if dt > (self.dx ** 2) / (4 * max_alpha):
            raise ValueError("CFL condition violated. Decrease time step or increase spatial resolution.")
        
        self.diffusion_map = self.create_diffusion_map()
        self.advection_map = self.create_advection_map()

    def create_diffusion_map(self):
        # interpolate alpha to the grid
        diffusion_map = torch.nn.functional.interpolate(self.alpha.view(1, 1, self.param_x, self.param_y), size=(self.N, self.N), mode='bilinear', align_corners=True)
        return diffusion_map.squeeze()
    
    def create_advection_map(self):
        # interpolate alpha to the grid
        advection_map = torch.nn.functional.interpolate(self.kappa.view(1, 1, self.param_x, self.param_y), size=(self.N, self.N), mode='bilinear', align_corners=True)
        return advection_map.squeeze()
    
    def penalization(self):
        return sum([torch.sum(torch.relu(-p)) for p in self.parameters()])

    def step(self, T):
        
        # t is not time, just time step
        dx2 = self.dx ** 2
        dx = self.dx

        # Compute updates for interior points
        T_xx = (T[:-2, 1:-1] - 2 * T[1:-1, 1:-1] + T[2:, 1:-1]) / dx2
        T_yy = (T[1:-1, :-2] - 2 * T[1:-1, 1:-1] + T[1:-1, 2:]) / dx2
        T_x = (T[2:, 1:-1] - T[:-2, 1:-1]) / (2 * dx)
        T_y = (T[1:-1, 2:] - T[1:-1, :-2]) / (2 * dx)
        
        alpha = self.diffusion_map[1:-1, 1:-1]
        kappa = self.advection_map[1:-1, 1:-1]

        # Compute the new temperature
        T_new = T.clone()
        

         # Burgers' term + Diffusion + Reaction
        T_new[1:-1, 1:-1] = (T[1:-1, 1:-1]
            - self.dt * kappa * (T[1:-1, 1:-1] * T_x + T[1:-1, 1:-1] * T_y)   # Advection (Burgers')
            + self.dt * alpha * (T_xx + T_yy)  # Diffusion
        )

        # Apply Neumann boundary conditions
        T_new[0, :] = T_new[1, :]
        T_new[-1, :] = T_new[-2, :]
        T_new[:, 0] = T_new[:, 1]
        T_new[:, -1] = T_new[:, -2]
        

        return T_new

    def forward(self, u0):
        """
        Simulate the heat equation over a number of time steps.
        """
        self.diffusion_map = self.create_diffusion_map()
        self.advection_map = self.create_advection_map()
        T = u0.clone().requires_grad_(True).to(self.device)  # Start with the initial condition
        intermediate_steps = torch.zeros(self.num_steps, self.N, self.N, device=self.device)

        for t in range(self.num_steps):
            intermediate_steps[t] = T.clone()  # Store the current state
            T = self.step(T)  # Compute the next state
        return intermediate_steps[1:]
    
    
    
################################################################################################
################# For 2 homogeneous materials with different properties ########################
################################################################################################

class HeatEquation2M(nn.Module):
    def __init__(self, device, L, N, dt, num_steps, material_boundary, k1=1.0, k2=1.0, alpha1=1.0, alpha2=1.0):
        super(HeatEquation2M, self).__init__()
        self.device = device
        self.L = L
        self.N = N
        
        self.dx = L / (N - 1)
        self.dt = dt
        self.num_steps = num_steps   
        
        # Placeholders
        self.nX = N
        self.param_x = 1    
        self.param_y = 1
        
        
        # Parameters with gradients
        self.k1 = nn.Parameter(torch.tensor(k1, dtype=torch.float32, device=self.device))
        self.k2 = nn.Parameter(torch.tensor(k2, dtype=torch.float32, device=self.device))
        self.alpha1 = nn.Parameter(torch.tensor(alpha1, dtype=torch.float32, device=self.device))
        self.alpha2 = nn.Parameter(torch.tensor(alpha2, dtype=torch.float32, device=self.device))
        
        self.material_boundary = material_boundary
        
        max_alpha = max(alpha1, alpha2)
        if dt > (self.dx ** 2) / (4 * max_alpha):
            raise ValueError("CFL condition violated. Decrease time step or increase spatial resolution.")
        
        self.material_map = self.create_material_map()
        self.k_map = self.create_conductivity_map()

    def create_material_map(self):
        material_map = torch.ones((self.N, self.N), dtype=torch.float32, device=self.device) * self.alpha2
        material_map[:, :self.material_boundary[1]] = self.alpha1
        return material_map

    def create_conductivity_map(self):
        k_map = torch.ones((self.N, self.N), dtype=torch.float32, device=self.device) * self.k2
        k_map[:, :self.material_boundary[1]] = self.k1
        return k_map
    
    def penalization(self):
        return sum([torch.relu(-p) for p in self.parameters()])

    def step(self, T):
        dx2 = self.dx ** 2

        # Compute updates for interior points
        T_xx = (T[:-2, 1:-1] - 2 * T[1:-1, 1:-1] + T[2:, 1:-1]) / dx2
        T_yy = (T[1:-1, :-2] - 2 * T[1:-1, 1:-1] + T[1:-1, 2:]) / dx2
        alpha = self.material_map[1:-1, 1:-1]

        # Compute the new temperature
        T_new = T.clone()
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + self.dt * alpha * (T_xx + T_yy)

        # Handle interface explicitly
        interface_idx = self.material_boundary[1] - 1
        k_left = self.k_map[:, interface_idx]
        k_right = self.k_map[:, interface_idx + 1]
        T_new[:, interface_idx] = (
            k_left * T[:, interface_idx + 1] + k_right * T[:, interface_idx - 1]
        ) / (k_left + k_right)

        # Apply Dirichlet boundary conditions
        T_new[0, :] = T_new[-1, :] = T_new[:, 0] = T_new[:, -1] = 0.0

        return T_new

    def forward(self, u0):
        """
        Simulate the heat equation over a number of time steps.
        """
        self.material_map = self.create_material_map()
        self.k_map = self.create_conductivity_map()
        T = u0.clone().requires_grad_(True).to(self.device)  # Start with the initial condition
        intermediate_steps = torch.zeros(self.num_steps, self.N, self.N, device=self.device)

        for t in range(self.num_steps):
            intermediate_steps[t] = T.clone()  # Store the current state
            T = self.step(T)  # Compute the next state
        return intermediate_steps[1:]

    
class AdvectionDiffusionReaction2M(nn.Module):
    def __init__(self, device, L, N, dt, num_steps, material_boundary, k1=1.0, k2=1.0, alpha1=1.0, alpha2=1.0):
        super(AdvectionDiffusionReaction2M, self).__init__()
        self.device = device
        self.L = L
        self.N = N
        
        self.dx = L / (N - 1)
        self.dt = dt
        self.num_steps = num_steps   
        
        # Placeholders
        self.nX = N
        self.param_x = 1    
        self.param_y = 1
        
        
        # Parameters with gradients
        self.k1 = nn.Parameter(torch.tensor(k1, dtype=torch.float32, device=self.device))
        self.k2 = nn.Parameter(torch.tensor(k2, dtype=torch.float32, device=self.device))
        self.alpha1 = nn.Parameter(torch.tensor(alpha1, dtype=torch.float32, device=self.device))
        self.alpha2 = nn.Parameter(torch.tensor(alpha2, dtype=torch.float32, device=self.device))
        
        self.material_boundary = material_boundary
        
        max_alpha = max(alpha1, alpha2)
        if dt > (self.dx ** 2) / (4 * max_alpha):
            print("CFL condition violated. Decrease time step or increase spatial resolution.")
        
        self.material_map = self.create_material_map()
        self.k_map = self.create_conductivity_map()

    def create_material_map(self):
        material_map = torch.ones((self.N, self.N), dtype=torch.float32, device=self.device) * self.alpha2
        material_map[:, :self.material_boundary[1]] = self.alpha1
        return material_map

    def create_conductivity_map(self):
        k_map = torch.ones((self.N, self.N), dtype=torch.float32, device=self.device) * self.k2
        k_map[:, :self.material_boundary[1]] = self.k1
        return k_map
    
    def penalization(self):
        return sum([torch.relu(-p) for p in self.parameters()])
    
    def stabilize(self):
        # Penalize to reduce reactivity and increase diffusion
        return 1e-3*(torch.nn.MSELoss()(self.k1, -self.k1) + torch.nn.MSELoss()(self.k2, -self.k2) + torch.nn.MSELoss()(self.alpha1, -self.alpha1) + torch.nn.MSELoss()(self.alpha2, -self.alpha2))
        

    def step(self, T, t):
        
        # t is not time, just time step
        dx2 = self.dx ** 2
        dx = self.dx

        # Compute updates for interior points
        T_xx = (T[:-2, 1:-1] - 2 * T[1:-1, 1:-1] + T[2:, 1:-1]) / dx2
        T_yy = (T[1:-1, :-2] - 2 * T[1:-1, 1:-1] + T[1:-1, 2:]) / dx2
        T_x = (T[2:, 1:-1] - T[:-2, 1:-1]) / (2 * dx)
        T_y = (T[1:-1, 2:] - T[1:-1, :-2]) / (2 * dx)
        
        alpha = self.material_map[1:-1, 1:-1]
        kappa = self.k_map[1:-1, 1:-1]

        # Compute the new temperature
        T_new = T.clone()
        

    # Burgers' term + Diffusion + Reaction
        T_new[1:-1, 1:-1] = (T[1:-1, 1:-1]
            - self.dt * (T[1:-1, 1:-1] ** 2 * T_x - T[1:-1, 1:-1] * T_y) * kappa  # Advection (Burgers')
            + self.dt * alpha * (T_xx + T_yy)  # Diffusion
            + self.dt * kappa * (T[1:-1, 1:-1]**3 - T[1:-1, 1:-1]**2 + T[1:-1, 1:-1])  # Reaction
    )
        # Handle interface explicitly
        interface_idx = self.material_boundary[1] - 1
        k_left = self.k_map[:, interface_idx]
        k_right = self.k_map[:, interface_idx + 1]
        T_new[:, interface_idx] = (
            k_left * T[:, interface_idx + 1] + k_right * T[:, interface_idx - 1]
        ) / (k_left + k_right)

        # Apply Neumann boundary conditions
        T_new[0, :] = T_new[1, :]
        T_new[-1, :] = T_new[-2, :]
        T_new[:, 0] = T_new[:, 1]
        T_new[:, -1] = T_new[:, -2]
        

        return T_new

    def forward(self, u0):
        """
        Simulate the heat equation over a number of time steps.
        """
        self.material_map = self.create_material_map()
        self.k_map = self.create_conductivity_map()
        T = u0.clone().requires_grad_(True).to(self.device)  # Start with the initial condition
        intermediate_steps = torch.zeros(self.num_steps, self.N, self.N, device=self.device)

        for t in range(self.num_steps):
            intermediate_steps[t] = T.clone()  # Store the current state
            T = self.step(T, t)  # Compute the next state
        return intermediate_steps[1:]
    
    
################################################################################################
################# For 1 homogeneous material, gaussian diffusivity/advectivity ########################
################################################################################################

class HeatEquationG(nn.Module):
    def __init__(self, device, L, N, dt, num_steps, num_gaussians, alpha = None):
        super(HeatEquationG, self).__init__()
        self.device = device
        self.L = L
        self.N = N
        
        self.dx = L / (N - 1)
        self.dt = dt
        self.num_steps = num_steps   
        
        # Placeholders
        self.nX = N
        self.num_params = 4 * num_gaussians # pos_x, pos_y, sigma, amplitude
        self.num_gaussians = num_gaussians
        
        # Parameters with gradients
        
        if alpha is not None:
            self.alpha = nn.Parameter(alpha)
            if self.alpha.shape != (self.num_params_alpha):
                raise ValueError("Shape of alpha does not match the parameters.")
        else:
            self.alpha = nn.Parameter(torch.ones((self.num_params), dtype=torch.float32, device=self.device))

        
        max_alpha = torch.max(self.alpha)
        if dt > (self.dx ** 2) / (4 * max_alpha):
            raise ValueError("CFL condition violated. Decrease time step or increase spatial resolution.")
        
        self.diffusion_map = self.create_diffusion_map()

    def create_diffusion_map(self):
        # interpolate alpha to the grid
        diffusion_map = torch.zeros((self.N, self.N), dtype=torch.float32, device=self.device)
        
        x = torch.linspace(-self.L//2, self.L//2, self.N)
        y = torch.linspace(-self.L//2, self.L//2, self.N)
        x, y = torch.meshgrid(x, y)
        for i in range(self.num_gaussians):
            diffusion_map += self.alpha[i] * torch.exp(-((x - self.alpha[self.num_gaussians + i]) ** 2 + (y - self.alpha[2*self.num_gaussians + i]) ** 2) / self.alpha[3*self.num_gaussians + i])
        return diffusion_map
    
    def penalization(self):
        return 0

    def step(self, T):
        dx2 = self.dx ** 2

        # Compute updates for interior points
        T_xx = (T[:-2, 1:-1] - 2 * T[1:-1, 1:-1] + T[2:, 1:-1]) / dx2
        T_yy = (T[1:-1, :-2] - 2 * T[1:-1, 1:-1] + T[1:-1, 2:]) / dx2
        alpha = self.diffusion_map[1:-1, 1:-1]

        # Compute the new temperature
        T_new = T.clone()
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + self.dt * alpha * (T_xx + T_yy)

        # Apply Dirichlet boundary conditions
        T_new[0, :] = T_new[-1, :] = T_new[:, 0] = T_new[:, -1] = 0.0

        return T_new

    def forward(self, u0):
        """
        Simulate the heat equation over a number of time steps.
        """
        self.diffusion_map = self.create_diffusion_map()
        T = u0.clone().requires_grad_(True).to(self.device)  # Start with the initial condition
        intermediate_steps = torch.zeros(self.num_steps, self.N, self.N, device=self.device)

        for t in range(self.num_steps):
            intermediate_steps[t] = T.clone()  # Store the current state
            T = self.step(T)  # Compute the next state
        return intermediate_steps[1:]
    
class AdvectionDiffusionG(nn.Module):
    def __init__(self, device, L, N, dt, num_steps, num_gaussians_alpha, num_gaussians_kappa, alpha = None, kappa = None):
        super(AdvectionDiffusionG, self).__init__()
        self.device = device
        self.L = L
        self.N = N
        
        self.dx = L / (N - 1)
        self.dt = dt
        self.num_steps = num_steps   
        
        # Placeholders
        self.nX = N
        self.num_params_alpha = 4 * num_gaussians_alpha # pos_x, pos_y, sigma, amplitude
        self.num_gaussians_alpha = num_gaussians_alpha
        self.num_params_kappa = 4 * num_gaussians_kappa # pos_x, pos_y, sigma, amplitude
        self.num_gaussians_kappa = num_gaussians_kappa
        
        # Parameters with gradients
        
        if alpha is not None:
            self.alpha = nn.Parameter(alpha)
            print(self.alpha.shape[0])
            print(self.num_params_alpha)
            if self.alpha.shape[0] != (self.num_params_alpha):
                raise ValueError("Shape of alpha does not match the parameters.")
        else:
            self.alpha = nn.Parameter(0.1*torch.randn((self.num_params_alpha), dtype=torch.float32, device=self.device) + 1)

        if kappa is not None:
            self.kappa = nn.Parameter(kappa)
            if self.kappa.shape[0] != (self.num_params_kappa):
                raise ValueError("Shape of kappa does not match the parameters.")
        else:
            self.kappa = nn.Parameter(0.1*torch.randn((self.num_params_alpha), dtype=torch.float32, device=self.device) + 1)
        
        print(self.alpha)
        print(self.kappa)
        max_alpha = torch.max(self.alpha)
        if dt > (self.dx ** 2) / (4 * max_alpha):
            raise ValueError("CFL condition violated. Decrease time step or increase spatial resolution.")
        
        self.diffusion_map = self.create_diffusion_map()

    def create_diffusion_map(self):
        # interpolate alpha to the grid
        diffusion_map = 0.1*torch.ones((self.N, self.N), dtype=torch.float32, device=self.device)
        
        x = torch.linspace(-self.L//2, self.L//2, self.N).to(self.device)
        y = torch.linspace(-self.L//2, self.L//2, self.N).to(self.device)
        x, y = torch.meshgrid(x, y, indexing='ij')
        for i in range(self.num_gaussians_alpha):
            diffusion_map += self.alpha[i] * torch.exp(-((x + self.alpha[self.num_gaussians_alpha + i]) ** 2 + (y + self.alpha[2*self.num_gaussians_alpha + i]) ** 2))
        return diffusion_map
    
    def create_advection_map(self):
        # interpolate alpha to the grid
        advection_map = 0.1*torch.ones((self.N, self.N), dtype=torch.float32, device=self.device)
        
        x = torch.linspace(-self.L//2, self.L//2, self.N).to(self.device)
        y = torch.linspace(-self.L//2, self.L//2, self.N).to(self.device)
        x, y = torch.meshgrid(x, y, indexing='ij')
        for i in range(self.num_gaussians_kappa):
            advection_map += self.kappa[i] * torch.exp(-((x + self.kappa[self.num_gaussians_kappa + i]) ** 2 + (y + self.kappa[2*self.num_gaussians_kappa + i]) ** 2))
        return advection_map
    
    def penalization(self):
        return 1e6*(torch.relu(-self.alpha[0]) + torch.relu(-self.kappa[0]))
    
    def stabilize(self):
        return 0.1*(torch.nn.MSELoss()(self.kappa[0], -self.kappa[0]) + torch.nn.MSELoss()(self.alpha[0], -self.alpha[0]))

    def step(self, T):
        
        # t is not time, just time step
        dx2 = self.dx ** 2
        dx = self.dx

        # Compute updates for interior points
        T_xx = (T[:-2, 1:-1] - 2 * T[1:-1, 1:-1] + T[2:, 1:-1]) / dx2
        T_yy = (T[1:-1, :-2] - 2 * T[1:-1, 1:-1] + T[1:-1, 2:]) / dx2
        T_x = (T[2:, 1:-1] - T[:-2, 1:-1]) / (2 * dx)
        T_y = (T[1:-1, 2:] - T[1:-1, :-2]) / (2 * dx)
        
        alpha = self.diffusion_map[1:-1, 1:-1]
        kappa = self.advection_map[1:-1, 1:-1]

        # Compute the new temperature
        T_new = T.clone()
        

         # Burgers' term + Diffusion + Reaction
        T_new[1:-1, 1:-1] = (T[1:-1, 1:-1]
            - self.dt * kappa * (T[1:-1, 1:-1] * T_x + T[1:-1, 1:-1] * T_y)   # Advection (Burgers')
            + self.dt * alpha * (T_xx + T_yy)  # Diffusion
        )

        # Apply Neumann boundary conditions
        T_new[0, :] = T_new[1, :]
        T_new[-1, :] = T_new[-2, :]
        T_new[:, 0] = T_new[:, 1]
        T_new[:, -1] = T_new[:, -2]
        

        return T_new

    def forward(self, u0):
        """
        Simulate the heat equation over a number of time steps.
        """
        self.diffusion_map = self.create_diffusion_map()
        self.advection_map = self.create_advection_map()
        T = u0.clone().requires_grad_(True).to(self.device)  # Start with the initial condition
        intermediate_steps = torch.zeros(self.num_steps, self.N, self.N, device=self.device)

        for t in range(self.num_steps):
            intermediate_steps[t] = T.clone()  # Store the current state
            T = self.step(T)  # Compute the next state
        return intermediate_steps[1:]
    
    
class WaveEquationG(nn.Module):
    def __init__(self, device, L, N, dt, num_steps, num_gaussians, alpha = None):
        super(WaveEquationG, self).__init__()
        self.device = device
        self.L = L
        self.N = N
        
        self.dx = L / (N - 1)
        self.dt = dt
        self.num_steps = num_steps   
        
        # Placeholders
        self.nX = N
        self.num_params = 4 * num_gaussians # pos_x, pos_y, sigma, amplitude
        self.num_gaussians = num_gaussians
        
        # Parameters with gradients
        
        if alpha is not None:
            self.alpha = nn.Parameter(alpha)
            if self.alpha.shape[0] != (self.num_params):
                raise ValueError("Shape of alpha does not match the parameters.")
        else:
            self.alpha = nn.Parameter(torch.ones((self.num_params), dtype=torch.float32, device=self.device))

        
        max_alpha = torch.max(self.alpha)
        if dt > (self.dx ** 2) / (4 * max_alpha):
            raise ValueError("CFL condition violated. Decrease time step or increase spatial resolution.")
        
        self.wave_speed_map = self.create_wave_speed_map()

    def create_wave_speed_map(self):
        # interpolate alpha to the grid
        wave_speed_map = 2*torch.ones((self.N, self.N), dtype=torch.float32, device=self.device)
        
        x = torch.linspace(-self.L//2, self.L//2, self.N)
        y = torch.linspace(-self.L//2, self.L//2, self.N)
        x, y = torch.meshgrid(x, y, indexing='ij')
        for i in range(self.num_gaussians):
            wave_speed_map += self.alpha[i] * torch.exp(-((x - self.alpha[self.num_gaussians + i]) ** 2 + (y - self.alpha[2*self.num_gaussians + i]) ** 2)) 
        return wave_speed_map
    
    def penalization(self):
        return 0

    def step(self, u, u_prev):
        dx2 = self.dx ** 2
        dt2 = self.dt ** 2
        c2 = self.wave_speed_map ** 2

        # Compute updates for interior points
        u_xx = (u[:-2, 1:-1] - 2 * u[1:-1, 1:-1] + u[2:, 1:-1]) / dx2
        u_yy = (u[1:-1, :-2] - 2 * u[1:-1, 1:-1] + u[1:-1, 2:]) / dx2
        c2_map = c2[1:-1, 1:-1]

        # Compute the new wave field using the wave equation update rule
        u_new = u.clone()
        u_new[1:-1, 1:-1] = (2 * u[1:-1, 1:-1] - u_prev[1:-1, 1:-1] +
                            dt2 * c2_map * (u_xx + u_yy))

        # Apply Dirichlet boundary conditions (zero displacement at the boundaries)
        u_new[0, :] = u_new[-1, :] = u_new[:, 0] = u_new[:, -1] = 0.0

        return u_new, u


    def forward(self, u0):
        """
        Simulate the heat equation over a number of time steps.
        """
        self.wave_speed_map = self.create_wave_speed_map()
        T = u0.clone().requires_grad_(True).to(self.device)  # Start with the initial condition
        T_prev = T.clone()
        intermediate_steps = torch.zeros(self.num_steps, self.N, self.N, device=self.device)

        for t in range(self.num_steps):
            intermediate_steps[t] = T.clone()  # Store the current state
            T, T_prev = self.step(T, T_prev)  # Compute the next state
            
        return intermediate_steps[1:]


import time

# Example Usage
if __name__ == "__main__":
    # Domain and discretization'
    device = torch.device("cpu")
    L = 6.0  # Length of the domain
    N = 50  # Grid size
    dt = 0.001  # Time step
    num_steps = 100

    # Actual material properties
    k1 = 1.5  # Conductivity of material 1
    k2 = 0.5  # Conductivity of material 2
    alpha1 = 0.5  # Diffusivity of material 1
    alpha2 = 0.15  # Diffusivity of material 2
    material_boundary = (0, N // 2)  # Split domain in half

    # Initialize model
    model_real = HeatEquation2M(device, L, N, dt, num_steps, material_boundary, k1, k2, alpha1, alpha2)

    # Set initial condition as gaussian
    u0 = torch.zeros(N, N, device=device)
    u0[N//4:3*N // 4, N//4:3*N // 4] = 1.0
    

    # Simulate
    t1 = time.time()
    actual = model_real(u0)
    t2 = time.time()
    print(t2-t1)
    
    # For inverse problem
    k1 = 1  # Conductivity of material 1
    k2 = 1  # Conductivity of material 2
    alpha1 = 1  # Diffusivity of material 1
    alpha2 = 1  # Diffusivity of material 2
    material_boundary = (0, N // 2)  # Split domain in half
    
    # Initialize model
    model = HeatEquation2M(device, L, N, dt, num_steps, material_boundary, k1, k2, alpha1, alpha2)

    # Set initial condition as gaussian
    u0 = torch.zeros(N, N)
    u0[N//4:3*N // 4, N//4:3*N // 4] = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        predicted = model(u0.detach())
        loss = criterion(actual.detach(), predicted)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}: Loss {loss.item()}")

    
    print("Optimization done")
    print(model.alpha1, model.alpha2, model.k1, model.k2)
    
    