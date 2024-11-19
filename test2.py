import torch
import torch.nn.functional as F

# Define the grid and the values (2D tensor)
grid_x = torch.linspace(0, 10, 5)  # X-coordinates
grid_y = torch.linspace(0, 10, 5)  # Y-coordinates
xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
values = torch.sin(xx) * torch.cos(yy)  # Example function on the grid

# Combine grid into a 2D array
grid_values = values.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

# Define the query point (x, y)
query_point = torch.tensor([[6.5, 3.5]])  # Example arbitrary point

# Normalize query point to [-1, 1] range
query_normalized = torch.stack((
    2 * (query_point[:, 0] - grid_x.min()) / (grid_x.max() - grid_x.min()) - 1,
    2 * (query_point[:, 1] - grid_y.min()) / (grid_y.max() - grid_y.min()) - 1
), dim=-1)

# Add batch and channel dimensions, and reshape
query_normalized = query_normalized.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 1, 2)

# Interpolation using grid_sample
interpolated = F.grid_sample(
    grid_values, query_normalized, mode='bilinear', align_corners=True
)

# Extract the interpolated value
interpolated_value = interpolated.squeeze().item()

print(f"The interpolated value at {query_point.tolist()} is {interpolated_value:.4f}")
