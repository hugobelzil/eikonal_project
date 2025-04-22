import numpy as np
from solvers.eikonal_general_F_2D import EikonalSolver
from solvers.ode_solver import ODE_backtracer
from computational_domain import ComputationalDomain
import matplotlib.pyplot as plt
import scienceplots
import cProfile

## BUILDING THE VELOCITY FUNCTION
def F(x,y):
    '''Function defining the velocity'''
    return np.where(y >= 0.0, 2, 1)



## BUILDING THE DOMAIN
dom = ComputationalDomain(N = 801, a = -1, b = 1, c = -1, d = 1)
dom.Gamma([(200,600)]) # Target when doing back tracing

#INSTANTIATING THE SOLVER
print("Solving Eikonal Equation on the domain...")
solver = EikonalSolver(domain = dom, F = F)
#cProfile.run('solver.SweepUntilConvergence()')

solver.SweepUntilConvergence(epsilon = 1e-6, verbose=True)
u = solver.grids_after_sweeps[-1]
np.save('snell_grid.npy', solver.grids_after_sweeps[-1])
# FINDING THE OPTIMAL PATH
x0 = [0,-0.5]
path = ODE_backtracer(x0 =x0,domain = dom, dt = dom.h/10, u_grid = u, tol = 1e-8, max_steps=150000)
print("First point of the path (cart. coords.) : ", path[0])
print("Last point of the path (cart. coords.) : ", path[-1])


print("Starting point in (x,y) : ",x0)

x = np.linspace(-1,1,801)
y = np.linspace(-1,1,801)
X,Y = np.meshgrid(x,y)

plt.style.use(['science', 'grid'])
plt.figure(figsize=(7.8, 4.8))
plt.imshow(u, origin = "upper", extent=(dom.a, dom.b, dom.c, dom.d), cmap='coolwarm')
plt.colorbar()
plt.plot(path[:,0], path[:,1], color='black', linewidth=1, label='Path')
plt.scatter(*x0, color='blue', s = 30, label=r'$x_0$')
plt.scatter(*[0.5,0.5], color='red', s = 30, label=r'$\Gamma$')
plt.legend()
plt.title(r"Optimal path backtraced solving $\partial_t X = -\nabla u(X)$", fontsize=18, pad = 10)
plt.xlabel("x")
plt.ylabel("y")# Optional: if your grid maps top-to-bottom
plt.tight_layout()
plt.savefig('../../plots/snell_path.png')
plt.show()


