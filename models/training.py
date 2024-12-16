import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np

import torch

def interpolate_phys_solution(data, phys_solution, nX=11):
    """
    Interpolate the phys solution onto the trajectory points for
    each time step using bilinear interpolation.

    Args:
    - data (tensor): A tensor of shape (T, N, 3)
                        containing the trajectories (x, y, u).
    - phys_solution (tensor): A tensor of shape (T, nX, nX)
                        containing the phys solution over time.

    Returns:
    - interpolated_solution (tensor): A tensor of shape (T, N)
                        containing the interpolated phys values
                        for each trajectory at each time step.
    """

    # Normalize query points across all time steps
    query_points = data[..., :2]  # Extract (x, y) coordinates
    query_normalized = (query_points + 3) / 3 - 1
    query_normalized = query_normalized.unsqueeze(1)

    # Prepare phys_solution for batch processing
    grid_values = phys_solution.unsqueeze(1)  # Shape: (T, 1, H, W)
    
    # Perform batch grid sampling
    interpolated = nn.functional.grid_sample(
        grid_values, query_normalized, mode='bilinear', align_corners=True
    )  # Output shape: (T, 1, N, 1)
    # Squeeze to match output shape
    interpolated_solution = interpolated.squeeze(1).squeeze(1)  # Shape: (T, N)

    return interpolated_solution



class Trainer:
    def __init__(self,
                 model_node,
                 model_phys,
                 optimizer_node,
                 optimizer_phys,
                 scheduler_anode,
                 scheduler_phys,
                 device,
                 grid,
                 print_freq=50, 
                 interaction = True):
        self.model_node = model_node.to(device)
        self.model_phys = model_phys.to(device)
        self.optimizer_node = optimizer_node
        self.optimizer_phys = optimizer_phys
        self.scheduler_anode = scheduler_anode
        self.scheduler_phys = scheduler_phys
        self.device = device
        self.loss_func = nn.MSELoss()
        self.print_freq = print_freq
        self.grid = grid
        self.interaction = interaction  

    def train(self, u, u0, num_epochs):
        u_train = (u[0, :, :].to(self.device))
        u_target = u[1:, :, :].to(self.device)
        self.start_time = time.time()
        if self.interaction:
            loss = 1
            epoch = 0
            #train trajectories first
            self.optimizer_node.param_groups[0]['lr'] = 1e-2
            while loss > 5e-3:
                loss = self._train_epoch_trajectory(u_train, u_target, u0)
                epoch += 1
                if (epoch + 1) % self.print_freq == 1:
                    elapsed_time = (time.time() - self.start_time) / 60.
                    print(
                        f"Epoch {epoch + 1} | Loss: {loss:.3e} | Time: {elapsed_time:.1f} min")

            # Train model
            self.optimizer_node.param_groups[0]['lr'] = 1e-3
            
            
            for epoch in range(num_epochs):
                loss, loss_FD = self._train_epoch(u_train, u_target, u0)
                if (epoch + 1) % self.print_freq == 1:
                    elapsed_time = (time.time() - self.start_time) / 60.
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss:.3e} | Loss_FD: {loss_FD:.3e}| Time: {elapsed_time:.1f} min | "
                        f"(alpha1, alpha2): {[x for x in self.model_phys.parameters()]}")

            print(f"Total time: {time.time() - self.start_time:.1f} sec")
            
        else:
            for epoch in range(num_epochs):
                loss= self._train_epoch_nointeraction(u_target, u0)
                if (epoch + 1) % self.print_freq == 1:
                    elapsed_time = (time.time() - self.start_time) / 60.
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss:.3e} | Time: {elapsed_time:.1f} min | "
                        f"(alpha1, alpha2): {[x for x in self.model_phys.parameters()]}")

            print(f"Total time: {time.time() - self.start_time:.1f} sec")

        return np.array([x.detach().cpu().numpy() for x in self.model_phys.parameters()])
            
        
    def _train_epoch_trajectory(self, u_train, u_target, u0):
        traj = self.model_node.g(u_train)
        
        
        # Calculate the loss
        loss = (100 * self.loss_func(traj, u_target[:,:,:2]))
        

        # Optimizer steps
        self.optimizer_node.zero_grad()
        loss.backward()
        self.optimizer_node.step()

        return loss.item()
    
    def _train_epoch_nointeraction(self, u_target, u0):
        # Compute phys solution using the current value of alpha
        phys_solution = self.model_phys(u0)

        interpolated_phys_target = interpolate_phys_solution(u_target, phys_solution)
        

        # Calculate the loss
        loss_FD = self.loss_func(interpolated_phys_target, u_target[:, :, 2])
        loss = 100 * loss_FD + 1e6 * self.model_phys.penalization()
        # Optimizer steps
        self.optimizer_phys.zero_grad()
        loss.backward()
        self.optimizer_phys.step()
        self.scheduler_phys.step()

        return loss.item()
        
        
        
    
    def _train_epoch(self, u_train, u_target, u0):
        # Compute phys solution using the current value of alpha
        phys_solution = self.model_phys(u0)

        # Compute trajectory predicted by NODE
        traj = self.model_node.f(u_train)

        # # Find node prediction on random points to compare with fd
        forward_random_points = torch.randn((10, 2)).to(self.device) -1
        u0_init = interpolate_phys_solution(forward_random_points.unsqueeze(0), u0.unsqueeze(0))
        init = torch.cat((forward_random_points, u0_init.T), dim=1)
        grid_traj_forward = self.model_node.f(init)

        # Interpolate phys at trajectories
        interpolated_phys_traj = interpolate_phys_solution(grid_traj_forward, phys_solution)
        
        interpolated_phys_target = interpolate_phys_solution(u_target, phys_solution)
        

        # Calculate the loss
        if interpolated_phys_target.all() < 1e2:
            loss_FD = self.loss_func(interpolated_phys_target, u_target[:, :, 2])
            loss_hybrid = self.loss_func(interpolated_phys_traj, grid_traj_forward[:,:,2])
        else:
            loss_FD = self.model_phys.stabilize()
            loss_hybrid = 0
        loss = (100 * self.loss_func(traj, u_target)
                 + self.model_phys.penalization()
                + 100 * loss_FD
                + loss_hybrid)
       

        # Optimizer steps
        self.optimizer_node.zero_grad()
        self.optimizer_phys.zero_grad()
        loss.backward()
        self.optimizer_node.step()
        self.optimizer_phys.step()
        self.scheduler_anode.step()
        self.scheduler_phys.step()

        return loss.item(), loss_FD.item()
