import numpy as np
from solvers.computational_domain import ComputationalDomain
from solvers.eikonal_general_F_2D import EikonalSolver

def F(x,y):
    return 1/np.sqrt((np.cos(x)**2)*(np.cos(y)**2)+(np.sin(x)**2)*(np.sin(y)**2))

N = 501
domain = ComputationalDomain(N = N, a=-2,b=2,c=-2,d=2)
domain.Gamma([(int((N-1)/2), int((N-1)/2))])

## SETTING BOUNDARY CONDITIONS
domain.grid[0,:] = np.cos(np.linspace(-2,2,N))*np.sin(2)
domain.grid[-1,:] = np.cos(np.linspace(-2,2,N))*np.sin(-2)
domain.grid[:,0] = np.sin(np.linspace(2,-2,N))*np.cos(-2)
domain.grid[:,-1] = np.sin(np.linspace(2,-2,N))*np.cos(2)

for i in range(N):
    domain.frozen.append((i,0))
    domain.frozen.append((i,N-1))
for j in range(N):
    domain.frozen.append((0,j))
    domain.frozen.append((N-1,j))

solver = EikonalSolver(domain, F = F)
print('Running convergence analysis')
solver.SweepUntilConvergence(epsilon=1e-5, verbose=True)
np.save('cosxsiny.npy',solver.grids_after_sweeps[-1])

