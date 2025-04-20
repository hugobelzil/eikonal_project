import numpy as np
from scipy.optimize import minimize_scalar
from computational_domain import ComputationalDomain
from solvers.ode_solver import ODE_backtracer

#SOLVING THE EXACT PATH

x0 = 0   #coordinates of starting point and target
y0 = -0.5
xT = 0.5
yT = 0.5

F1 = 1 #speed in the media (refraction index)
F2 = 1

def travel_time(x):
    a = np.sqrt((x-x0)**2 + y0**2)/F1
    b = np.sqrt((x-xT)**2 + yT**2)/F2
    return a+b

opt = minimize_scalar(travel_time, bounds = (-1,1), method = 'bounded')
x_opt = opt.x

segment1 = np.linspace([x0, y0],[x_opt,0], 100000)
segment2 =  np.linspace([x_opt, 0], [xT, yT], 100000)
true_path = np.vstack([segment1, segment2])

#LOADING THE GRID & REPLICATING THE VARIABLES IN exampleSnell.py
dom = ComputationalDomain(N = 801, a = -1, b = 1, c = -1, d = 1)
dom.Gamma([(200,600)])
grid = np.load('snell_grid.npy')

# COMPUTING THE PATH
path = ODE_backtracer(x0 = [x0,y0], dt = dom.h, domain = dom, u_grid = grid)

print(path[0],path[1], path[-1])
print(true_path[0], true_path[1], true_path[-1])

l = [i for i in range(10,0,-1)]
for j in range(2,11):
    l.append(float(1/j))

for f in l:
