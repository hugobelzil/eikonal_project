import numpy as np
from solvers.eikonal_general_F_2D import EikonalSolver
from solvers.ode_solver import ODE_backtracer
from computational_domain import ComputationalDomain
import matplotlib.pyplot as plt

### DOMAIN
t = np.pi/2
domain = ComputationalDomain(N=801,a=-t,b=t,c=-t,d=t)
domain.Gamma([(101,101)])

### VELOCITY

np.random.seed(42)
values = 5 * np.random.rand(5, 5)


def F(x, y):
    x_normalized = (x + np.pi / 2) / np.pi
    y_normalized = (y + np.pi / 2) / np.pi

    i = int(np.clip(y_normalized * 5, 0, 4 - 1e-8))
    j = int(np.clip(x_normalized * 5, 0, 4 - 1e-8))

    return values[i, j]


### SOLVER
solver = EikonalSolver(domain = domain, F = F)
solver.SweepUntilConvergence(epsilon=1e-6)
grid = solver.grids_after_sweeps[-1]

### 3D PLOT OF U
X,Y = np.meshgrid(np.linspace(-t,t,801),np.linspace(-t,t,801))
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, np.flipud(grid), cmap='jet', edgecolor='none', alpha=0.9)

# Formatting
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_zlabel('u(x,y)', fontsize=14)
ax.set_title(r'Surface plot of the solution $u(x,y)$', fontsize=16)
ax.view_init(elev=45, azim=135)  # Optional: adjust viewing angle
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.tight_layout()
plt.show()


### PATH BACKTRACING
backtraced = ODE_backtracer(x0 = [t-0.09,-t + 0.09], dt = 2*domain.h, domain=domain, u_grid=grid)

### PLOTTING THE BACKTRACED PATH
Z = np.vectorize(F)(X,Y)

plt.imshow(Z, origin = 'lower', extent = (-t,t,-t,t), cmap='jet')
plt.scatter(backtraced[:,0],backtraced[:,1],color='black', linewidth=0.5, label='Path')
plt.scatter(*[-t+101*domain.h,t-101*domain.h], color='red', marker='o', label='target')
plt.scatter(*[t-0.09,0.09], color = 'blue', marker = 'o',label = 'x0')
plt.title('backtraced path')
plt.show()