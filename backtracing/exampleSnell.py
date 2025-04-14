import numpy as np

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
dom.Gamma([(19, 131)]) # Target when doing back tracing

#INSTANTIATING THE SOLVER
solver = EikonalSolver(domain = dom, F = F)
solver.SweepUntilConvergence(epsilon = 1e-5, verbose=True)

#SAVING THE FUNCTION
np.save('snell_grid.npy',solver.grids_after_sweeps[-1])