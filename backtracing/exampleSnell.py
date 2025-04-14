import numpy as np
from interpolation.interpolator import bilinear_gradient
from solvers.eikonal_general_F_2D import EikonalSolver
from computational_domain import ComputationalDomain


## BUILDING THE VELOCITY FUNCTION
def F(x,y):
    '''Function defining the velocity'''
    if x >= 0.5:
        return 2
    elif (x < 0.5) and (x >= 0.25):
        return 0.1
    return 1

## BUILDING THE DOMAIN
dom = ComputationalDomain(N = 151, a = 0, b = 1, c = 0, d = 1)
dom.Gamma([(10,10)]) # Target when doing back tracing

#INSTANTIATING THE SOLVER
solver = EikonalSolver(domain = dom, F = F)
solver.SweepUntilConvergence(epsilon = 1e-5, verbose=True)
test = solver.grids_after_sweeps[-1]

#SAVING THE FUNCTION
def ODE_backtracer(x0, dt, domain, u_grid, max_steps=100000, tol=1e-10):
    """
    Trace a characteristic backward from point x0 using ∇u and Euler's method.
    Assumes u_grid is 2D array of u values and x0 is in Cartesian coordinates.
    """
    path = [np.array(x0)]
    x, y = x0

    for _ in range(max_steps):
        grad = bilinear_gradient(u_grid, x, y, domain)
        norm = np.linalg.norm(grad)

        if norm < tol:
            break

        # Euler step along -∇u direction (normalized)
        x -= dt * grad[0] / norm
        y -= dt * grad[1] / norm

        # Stop if outside domain
        if not (domain.a <= x <= domain.b and domain.c <= y <= domain.d):
            break

        path.append(np.array([x, y]))

    return np.array(path)


p = ODE_backtracer([0.85,0.1],domain = dom, dt = dom.h/2, u_grid=test)
print(p[0],p[1],p[-1],p[-1])


import matplotlib.pyplot as plt
target = np.array([dom.a + 131 * dom.h, dom.d - 19 * dom.h])
print("Target in (x,y):", target)

plt.imshow(test,origin = "lower", extent=[dom.a, dom.b, dom.c, dom.d])
plt.plot(p[:,0], p[:,1], color='black', linewidth=2, label="Backtraced path")
plt.scatter(*target, color='red', s=50, label="Gamma point")
plt.legend()
plt.title("Backtraced Characteristic")
plt.xlabel("x")
plt.ylabel("y")# Optional: if your grid maps top-to-bottom
plt.show()
