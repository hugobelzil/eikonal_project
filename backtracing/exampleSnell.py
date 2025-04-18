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
    if x >= 0.5:
        return 2
    elif (x < 0.5) and (x >= 0.25):
        return 0.1
    return 1

def F(x,y):
    if ((x-0.5)**2)+((y-0.5)**2)<=1/10:
        return 0.2
    return 1

def F(x,y):
    '''Function defining the velocity'''
    if x >= 0.75:
        return 1
    elif (x < 0.75) and (x >= 0.60):
        return 0.5
    elif (x < 0.60) and (x >= 0.40):
        return 1.5
    elif (x < 0.40) and (x >= 0.15):
        return 0.3
    return 1

## BUILDING THE DOMAIN
dom = ComputationalDomain(N = 151, a = 0, b = 1, c = 0, d = 1)
dom.Gamma([(10,10)]) # Target when doing back tracing

#INSTANTIATING THE SOLVER
print("Solving Eikonal Equation on the domain...")
solver = EikonalSolver(domain = dom, F = F)
#cProfile.run('solver.SweepUntilConvergence()')

solver.SweepUntilConvergence(epsilon = 1e-5, verbose=True)
u = solver.grids_after_sweeps[-1]

# FINDING THE OPTIMAL PATH
x0 = [0.85,0.1]
path = ODE_backtracer(x0 =x0,domain = dom, dt = dom.h/5, u_grid = u, tol = 1e-8, max_steps=150000)
print("First point of the path (cart. coords.) : ", path[0])
print("Last point of the path (cart. coords.) : ", path[-1])


target = np.array([dom.a + 10*dom.h, dom.d - 10*dom.h])
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



## 3D PLOT OF THE SOLUTION U
x = np.linspace(0,1,151)
y = np.linspace(0,1,151)
X,Y = np.meshgrid(x,y)
Z = u

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none')

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$u(x, y)$')
ax.set_title('Numerical Solution on $[-1,1]^{2}$, with $N=301$, 5 random points in $\Gamma$', fontsize = 18)
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show(block=True)
