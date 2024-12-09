# plot gaussians with given parameters

import numpy as np
import matplotlib.pyplot as plt


param_1 = [1, 1,  2, 1.0]
param_2 = [1, -1, -2, 1.0]
param_hybrid_1 = [0.7626, 0.9252, 1.8222, 1.0000]
param_hybrid_2 = [ 0.1609,  1.4781, -1.1994,  1.0000]
param_fd_1 = [0.2870, 0.5431, 0.9248, 0.9243]
param_fd_2 = [ 8.2319, -1.2583,  0.5696,  1.0000]


def gaussian(param, num_gaussians=1, N=100, L=6):
    # interpolate alpha to the grid
    advection_map = 0.1*np.ones((N, N), dtype=np.float32)
    
    x = np.linspace(-L//2, L//2, N)
    y = np.linspace(-L//2, L//2, N)
    x, y = np.meshgrid(x, y, indexing='ij')
    for i in range(num_gaussians):
        advection_map += param[i] * np.exp(-((x - param[num_gaussians + i]) ** 2 + (y - param[2*num_gaussians + i]) ** 2))
    return advection_map

z_1 = gaussian(param_1)
z_hybrid_1 = gaussian(param_hybrid_1, num_gaussians=1)
z_fd_1 = gaussian(param_fd_1, num_gaussians=1)
z_2 = gaussian(param_2)
z_hybrid_2 = gaussian(param_hybrid_2)
z_fd_2 = gaussian(param_fd_2)


fig = plt.figure()
ax = fig.subplots(1, 3)



ax[0].imshow(z_fd_1 + z_fd_2, cmap='coolwarm', extent=[-3, 3, -3, 3])
ax[0].set_title('Alpha Physics')

ax[1].imshow(z_hybrid_1 + z_hybrid_2, cmap='coolwarm', extent=[-3, 3, -3, 3])
ax[1].set_title('Alpha Hybrid')


ax[2].imshow(z_1 + z_2, cmap='coolwarm', extent=[-3, 3, -3, 3])
ax[2].set_title('Alpha Real')
plt.show()
