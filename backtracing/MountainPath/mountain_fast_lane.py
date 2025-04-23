from computational_domain import ComputationalDomain
from solvers.ode_solver import ODE_backtracer
from solvers.eikonal_general_F_2D import EikonalSolver
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

def F(x,y):
    if (x-0.8)**2 + (y-0.3)**2 <= 0.01:
        return 0.25
    if (x-0.3)**2+(y-0.5)**2<=0.04:
        return 0.2
    if (x-0.25)**2+(y-0.85)**2<=0.01:
        return 0.1
    if (x<=0.65 and x>=0.55) and (y>=0.35 and y<=0.55):
        return 0.08
    if (x>=0.05 and x<=0.1) and (y<=0.6 and y>=0.1):
        return 6
    #if (x>=0.05 and x<=0.5) and ()
    return 1

dom = ComputationalDomain(N=401, a = 0, b = 1, c = 0, d = 1)
dom.Gamma([(40, 40)])

solver = EikonalSolver(domain = dom, F = F)
solver.SweepUntilConvergence(epsilon = 1e-5, verbose=True)
u = solver.grids_after_sweeps[-1]

xf = np.linspace(0,1,401)
yf = np.linspace(0,1,401)
X,Y = np.meshgrid(xf,yf)
Z = np.vectorize(F)(X,Y)

# FINDING THE OPTIMAL PATH
x0 = [0.9,0.1]
path = ODE_backtracer(x0 =x0,domain = dom, dt = dom.h/5, u_grid = u, tol = 1e-8, max_steps=150000)
print("First point of the path (cart. coords.) : ", path[0])
print("Last point of the path (cart. coords.) : ", path[-1])


target = np.array([dom.a + 40*dom.h, dom.d - 40*dom.h])
print("Starting pint in (x,y) : ",x0)
print("Target in (x,y) : ", target)

plt.style.use(['science', 'grid'])
plt.figure(figsize=(7.8, 4.8))
# if displaying the function F, origin must be 'lower'. If showing u : origin = 'upper'
plt.imshow(Z, origin = "lower", extent=(dom.a, dom.b, dom.c, dom.d), cmap='plasma')
cbar = plt.colorbar()
cbar.set_label(r"$F(x, y)$", fontsize=14)
plt.plot(path[:,0], path[:,1], color='black', linewidth=1, label='Path')
plt.scatter(*target, color='red', s=30, label=r'$\Gamma$')
plt.scatter(*x0, color='blue', s = 30, label=r'$x_0$')
plt.legend()
plt.title(r"Optimal path backtraced solving $\partial_t X = -\nabla u(X)$, with a fast lane", fontsize=16, pad = 10)
plt.xlabel("x")
plt.ylabel("y")# Optional: if your grid maps top-to-bottom
plt.tight_layout()
plt.savefig('../../plots/path_mountain_fast_lane.png', dpi = 600)
plt.show()

#### PLOT OF THE SURFACE OF U
x = np.linspace(dom.a, dom.b, dom.N)
y = np.linspace(dom.c, dom.d, dom.N)
X, Y = np.meshgrid(x, y)

# Plot the surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, np.flipud(u), cmap='jet', edgecolor='none', alpha=0.9)

# Formatting
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.set_zlabel('u(x,y)', fontsize=14)
ax.set_title(r'Surface plot of the solution $u(x,y)$ with a fast lane', fontsize=18)
ax.view_init(elev=45, azim=135)  # Optional: adjust viewing angle
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.tight_layout()
plt.savefig('../../plots/surface_mountain_fast_lane.png', dpi = 600)
plt.show()