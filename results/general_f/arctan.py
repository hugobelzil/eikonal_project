import numpy as np
from eikonal_general_F_2D import EikonalSolver
from computational_domain import ComputationalDomain

test = ComputationalDomain(N=601, a=-1, b=1, c=-1, d=1)
test.Gamma([(300, 300)])

def F(x, y):
    return 1 + 0.5*(x**2 + y**2)


solver = EikonalSolver(test, F)
solver.SweepUntilConvergence(epsilon = 1e-5, verbose = True)
print("Exact value in the top left corner : ", np.sqrt(2)*np.arctan(1))
print("check : top left corner value sweep at gamma : ", solver.grids_after_sweeps[-1][0,0])
np.save('arctan_grid.npy', solver.grids_after_sweeps[-1])