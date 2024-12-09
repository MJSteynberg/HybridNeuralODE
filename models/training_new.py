import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

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

    # Normalize query points across all time steps
    query_points = data[..., :2]  # Extract (x, y) coordinates
    query_normalized = (query_points + 3) / 3 - 1
    query_normalized = query_normalized.unsqueeze(1)

    # Prepare heat_solution for batch processing
    grid_values = heat_solution.unsqueeze(1)  # Shape: (T, 1, H, W)
    

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
                 model_heat,
                 optimizer_node,
                 optimizer_heat,
                 scheduler_anode,
                 scheduler_heat,
                 device,
                 grid,
                 reg='l1',
                 print_freq=50, 
                 interaction = True):
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
                        f"(alpha1, alpha2): {[x for x in self.model_heat.parameters()]}")

            print(f"Total time: {time.time() - self.start_time:.1f} sec")
            
        else:
            for epoch in range(num_epochs):
                loss= self._train_epoch_nointeraction(u_target, u0)
                if (epoch + 1) % self.print_freq == 1:
                    elapsed_time = (time.time() - self.start_time) / 60.
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss:.3e} | Time: {elapsed_time:.1f} min | "
                        f"(alpha1, alpha2): {[x for x in self.model_heat.parameters()]}")

            print(f"Total time: {time.time() - self.start_time:.1f} sec")
            
        
    def _train_epoch_trajectory(self, u_train, u_target, u0):
        traj = self.model_node.g(u_train)
        
        # Regularization parameter and calculation
        lambda_reg = 1e-4 if self.reg == 'l1' else 1e-3
        ls_reg = 0.0
        # plt.clf()
        # for i in range(traj.shape[1]):
        #     plt.plot(traj[:, i, 0].detach().cpu().numpy(), traj[:, i, 1].detach().cpu().numpy(), 'r')
        #     plt.plot(u_target[:, i, 0].detach().cpu().numpy(), u_target[:, i, 1].detach().cpu().numpy(), 'b')
        # plt.pause(0.1)

        # if self.reg == 'l1':
        #     # Use list comprehension to avoid redundant attribute calls
        #     weights = [self.model_node.dynamics.fc1_time_1[:].weight,
        #                self.model_node.dynamics.fc1_time_2.weight,
        #                self.model_node.dynamics.fc3_time_1.weight,
        #                self.model_node.dynamics.fc3_time_2.weight]
        #     ls_reg = sum(w.abs().sum() for w in weights)

        # elif self.reg == 'barron':
        #     # Optimized Barron regularization calculation
        #     weights_fc1 = torch.cat([self.model_node.dynamics.fc1_time_1.weight,
        #                              self.model_node.dynamics.fc1_time_2.weight], dim=0)
        #     weights_fc3 = torch.cat([self.model_node.dynamics.fc3_time_1.weight,
        #                              self.model_node.dynamics.fc3_time_2.weight], dim=0)
        #     ls_reg = (weights_fc1.pow(2).sum(dim=1).sqrt() *
        #               weights_fc3.pow(2).sum(dim=1).sqrt().sum())

        # Calculate the loss
        loss = (100 * self.loss_func(traj, u_target[:,:,:2]))
                #+ lambda_reg * ls_reg)
        

        # Optimizer steps
        self.optimizer_node.zero_grad()
        loss.backward()
        self.optimizer_node.step()

        return loss.item()
    
    def _train_epoch_nointeraction(self, u_target, u0):
        # Compute heat solution using the current value of alpha
        heat_solution = self.model_heat(u0)

        interpolated_heat_target = interpolate_heat_solution(u_target, heat_solution)
        

        # Calculate the loss
        loss_FD = self.loss_func(interpolated_heat_target, u_target[:, :, 2])
        loss = 100 * loss_FD + 1e6 * self.model_heat.penalization()
        # Optimizer steps
        self.optimizer_heat.zero_grad()
        loss.backward()
        self.optimizer_heat.step()
        self.scheduler_heat.step()

        return loss.item()
        
        
        
    
    def _train_epoch(self, u_train, u_target, u0):
        # Compute heat solution using the current value of alpha
        heat_solution = self.model_heat(u0)

        # Compute trajectory predicted by NODE
        traj = self.model_node.f(u_train)

        # # Find node prediction on random points to compare with fd
        forward_random_points = torch.randn((8, 2)).to(self.device)
        u0_init = interpolate_heat_solution(forward_random_points.unsqueeze(0), u0.unsqueeze(0))
        init = torch.cat((forward_random_points, u0_init.T), dim=1)
        grid_traj_forward = self.model_node.f(init)

        # Interpolate heat at trajectories
        interpolated_heat_traj = interpolate_heat_solution(grid_traj_forward, heat_solution)
        
        interpolated_heat_target = interpolate_heat_solution(u_target, heat_solution)
        
        

        # Regularization parameter and calculation
        lambda_reg = 1e-4 if self.reg == 'l1' else 1e-3
        ls_reg = 0.0

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
        if interpolated_heat_target.all() < 1e2:
            loss_FD = self.loss_func(interpolated_heat_target, u_target[:, :, 2])
            loss_hybrid = self.loss_func(interpolated_heat_traj, grid_traj_forward[:,:,2])
        else:
            loss_FD = self.model_heat.stabilize()
            loss_hybrid = 0
        loss = (100 * self.loss_func(traj, u_target)
                 + self.model_heat.penalization()
                + 100 * loss_FD
                + loss_hybrid)
       

        # Optimizer steps
        self.optimizer_node.zero_grad()
        self.optimizer_heat.zero_grad()
        loss.backward()
        self.optimizer_node.step()
        self.optimizer_heat.step()
        self.scheduler_anode.step()
        self.scheduler_heat.step()

        return loss.item(), loss_FD.item()
