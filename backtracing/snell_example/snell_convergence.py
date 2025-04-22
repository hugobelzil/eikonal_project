import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import directed_hausdorff
from computational_domain import ComputationalDomain
from solvers.ode_solver import ODE_backtracer
import matplotlib.pyplot as plt

#SOLVING THE EXACT PATH

x0 = 0   #coordinates of starting point and target
y0 = -0.5
xT = 0.5
yT = 0.5

F1 = 1 #speed in the media (refraction index)
F2 = 2

def travel_time(x):
    a = np.sqrt((x-x0)**2 + y0**2)/F1
    b = np.sqrt((x-xT)**2 + yT**2)/F2
    return a+b

opt = minimize_scalar(travel_time, bounds = (-1,1), method = 'bounded')
x_opt = opt.x

segment1 = np.linspace([x0, y0],[x_opt,0], 50000)
segment2 =  np.linspace([x_opt, 0], [xT, yT], 50000)
true_path = np.vstack([segment1, segment2])

#LOADING THE GRID & REPLICATING THE VARIABLES IN exampleSnell.py
dom = ComputationalDomain(N = 801, a = -1, b = 1, c = -1, d = 1)
dom.Gamma([(200,600)])
grid = np.load('../snell_grid.npy')

# COMPUTING THE PATH
path = ODE_backtracer(x0 = [x0,y0], dt = dom.h, domain = dom, u_grid = grid)


l = [i for i in range(10,0,-1)]
for j in range(2,11):
    l.append(float(1/j))
h_distances = []
for k in l:
    print('k = {}'.format(k))
    path = ODE_backtracer(x0 = [x0,y0], dt = dom.h*k, domain = dom, u_grid = grid)
    d1 = directed_hausdorff(path, true_path)[0]
    d2 = directed_hausdorff(true_path, path)[0]
    hausdorff_distance = max(d1, d2)
    h_distances.append(hausdorff_distance)
    print(hausdorff_distance)


# Convert `l` to an array of step sizes
dt_values = np.array([dom.h * k for k in l])
h_distances = np.array(h_distances)

# Sort by increasing dt (for clean plotting)
sorted_indices = np.argsort(dt_values)
dt_values = dt_values[sorted_indices]
h_distances = h_distances[sorted_indices]

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(dt_values, h_distances, lw=2, label='Hausdorff distance')
plt.xlabel("Step size (dt)")
plt.ylabel("Hausdorff distance")
plt.title("Convergence of Backtraced Path vs True Path")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.show()