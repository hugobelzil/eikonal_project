import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import directed_hausdorff
from solvers.computational_domain import ComputationalDomain
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
grid = np.load('snell_grid.npy')
print('h : ',dom.h)
# COMPUTING THE PATH
path = ODE_backtracer(x0 = [x0,y0], dt = dom.h, domain = dom, u_grid = grid)


dt = np.linspace(3 * dom.h, dom.h, 35)
h_distances = []

for dt_value in dt:
    print('path planning with dt=',dt_value)
    path = ODE_backtracer(x0 = [x0,y0], dt = dt_value, domain = dom, u_grid = grid)
    d1 = directed_hausdorff(path, true_path)[0]
    d2 = directed_hausdorff(true_path, path)[0]
    hausdorff_distance = max(d1, d2)
    h_distances.append(hausdorff_distance)
    print(hausdorff_distance)


h_distances = np.array(h_distances)
np.save('dt_values_larger_window.npy',np.array(dt))
np.save('haussdorf_distances_larger_window.npy',h_distances)


plt.show()