from computational_domain import ComputationalDomain
from solvers.ode_solver import ODE_backtracer
from solvers.eikonal_general_F_2D import EikonalSolver
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

def F(x,y):
    if (x-0.8)**2 + (y-0.3)**2 <= 0.01:
        return 0.1
    if (x-0.3)**2+(y-0.5)**2<=0.04:
        return 0.1
    if (x-0.25)**2+(y-0.85)**2<=0.01:
        return 0.1
    return 1

dom = ComputationalDomain(N=200, a = 0, b = 1, c = 0, d = 1)
dom.Gamma([(20, 20)])

solver = EikonalSolver(domain = dom, F = F)
solver.SweepUntilConvergence(epsilon = 1e-5, verbose=True)
u = solver.grids_after_sweeps[-1]

# FINDING THE OPTIMAL PATH
x0 = [0.9,0.1]
path = ODE_backtracer(x0 =x0,domain = dom, dt = dom.h/5, u_grid = u, tol = 1e-8, max_steps=150000)
print("First point of the path (cart. coords.) : ", path[0])
print("Last point of the path (cart. coords.) : ", path[-1])


target = np.array([dom.a + 20*dom.h, dom.d - 20*dom.h])
print("Starting pint in (x,y) : ",x0)
print("Target in (x,y) : ", target)

plt.style.use(['science', 'grid'])
plt.figure(figsize=(7.8, 4.8))
plt.imshow(u, origin = "upper", extent=(dom.a, dom.b, dom.c, dom.d), cmap='jet')
plt.colorbar()
plt.plot(path[:,0], path[:,1], color='black', linewidth=1, label='Path')
plt.scatter(*target, color='red', s=30, label=r'$\Gamma$')
plt.scatter(*x0, color='blue', s = 30, label=r'$x_0$')
plt.legend()
plt.title(r"Optimal path backtraced solving $\partial_t X = -\nabla u(X)$", fontsize=18, pad = 10)
plt.xlabel("x")
plt.ylabel("y")# Optional: if your grid maps top-to-bottom
plt.tight_layout()
plt.show()