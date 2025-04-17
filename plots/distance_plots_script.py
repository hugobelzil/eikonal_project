import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from sklearn.linear_model import LinearRegression

grid = np.load('../results/distance/2D_distance_grid.npy')
errors = np.load('../results/distance/2D_distance_errors.npy')
index = np.load('../results/distance/N_index_errors.npy')

# Convergence plot for error vs. grid size N
plt.style.use(['science', 'grid'])
plt.figure(figsize=(7.8, 4.8))
plt.plot(index, errors,  marker='x', label=r'$L^{\infty}$ Error')
plt.xlabel("Grid size (N)")
plt.ylabel("Max Error")
plt.yscale("log")
plt.title(r'Convergence of Eikonal Solver for distance function on $[-1,1]^2$, $\epsilon = 10^{-6}$')
plt.legend()
plt.tight_layout()
plt.savefig('convergence2d.png', dpi=600, bbox_inches='tight')
#plt.show(block=True)  # Keep the window open

# CONVERGENCE PLOT OF ERROR VERSUS STEP SIZE h

#Start with the linear model
model = LinearRegression()
X = np.array([(2/(N-1)) for N in index]).reshape(-1,1)
model.fit(np.abs(X*np.log(X)), errors)
A = float(model.coef_)
B = float(model.intercept_)
x = np.linspace(np.min(X),np.max(X), 150)


# Plot
plt.figure(figsize=(7.8, 4.8))
plt.plot([(2/(N-1)) for N in index][::-1], np.flip(errors),  marker='x', label=r'$L^{\infty}$ Error')
plt.plot(x, A*np.abs(x*np.log(x)) + B, label = fr'${round(A,4)} \cdot |h \log h| + {round(B,4)}$')
plt.xlabel(r'Step size $h$')
plt.ylabel("Max Error")
plt.yscale("log")
plt.title(r'Convergence of Eikonal Solver for distance function on $[-1,1]^2$, $\epsilon = 10^{-6}$')
plt.legend()
plt.tight_layout()
plt.savefig('convergence2d_step_size.png', dpi=600, bbox_inches='tight')
#plt.show(block=True)

# 3D PLOT FOR THE SOLUTION

x = np.linspace(-1, 1, 257)
y = np.linspace(-1, 1, 257)
X, Y = np.meshgrid(x, y)

# 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, grid, cmap='jet', edgecolor='none')

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$u(x, y)$')
ax.set_title('Numerical Solution on $[-1,1]^{2}$, with $N=257$, $h=0.0078$', fontsize = 18)
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.tight_layout()
plt.savefig('solution_3d_surface.png', dpi=600)
plt.show(block=True)

# CONTOUR PLOT OF THE SOLUTION
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, grid, levels = 15, cmap = 'jet')
plt.colorbar(contour)

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Contours of the numerical solution over the square $[-1,1]^{2}$', fontsize = 18, pad = 20)
plt.tight_layout()
plt.savefig('contour_plot_2D_distance.png', dpi=600, bbox_inches='tight')
plt.show()

## PLOTS FOR RANDOM STENCILS

random5 = np.load('../results/random_stencils_distance/random5stencils_grid.npy')
random30 = np.load('../results/random_stencils_distance/random30stencils_grid.npy')

x = np.linspace(-1, 1, 301)
y = np.linspace(-1, 1, 301)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, random5, cmap='jet', edgecolor='none')

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$u(x, y)$')
ax.set_title('Numerical Solution on $[-1,1]^{2}$, with $N=301$, 5 random points in $\Gamma$', fontsize = 18)
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.tight_layout()
plt.savefig('solution_3d_surface_random5.png', dpi=600)
plt.show(block=True)

# CONTOUR PLOT FOR THE FUNCTION ABOVE
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, random5, levels = 15, cmap = 'jet')
plt.colorbar(contour)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Contours of the numerical solution over the square $[-1,1]^{2}$', fontsize = 18, pad  = 20)
plt.tight_layout()
plt.savefig('contour_plot_random5.png', dpi=600, bbox_inches='tight')
plt.show()

# HEATMAP OF THE SOLUTION WITH 30 RANDOM STENCILS
plt.figure(figsize=(8, 6))
plt.imshow(random30, cmap='jet', extent=(-1, 1, -1, 1))
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.colorbar()
plt.title('Heat map for $u(x,y)$ on $[-1,1]^{2}$, $N=301$, $30$ random points in $\Gamma$', fontsize = 18, pad = 20)
plt.tight_layout()
plt.savefig('heatmap_random30.png', dpi=600, bbox_inches='tight')
plt.show()
print("ALL PLOTS HAVE BEEN GRAPHED")