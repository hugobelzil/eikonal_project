import numpy as np
from eikonal_distance_2D import EikonalSolver
from computational_domain import ComputationalDomain

# BUILDING THE UNIT SQUARE
n = 256
unit_square = ComputationalDomain(N = n+1, a = -1, b = 1, c = -1, d = 1)
unit_square.Gamma([(int(n/2),int(n/2))])

# CREATE AN INSTANCE OF THE SOLVER
solver = EikonalSolver(unit_square)
solver.SweepUntilConvergence(epsilon = 1e-6, verbose = True)
np.save('2D_distance_grid.npy', solver.grids_after_sweeps[-1])
print("PDE solved on unit square grid")
