import numpy as np
from computational_domain import ComputationalDomain
from eikonal_distance_2D import EikonalSolver
import matplotlib.pyplot as plt
np.random.seed(42)

N = 301 # for grid size

# THE EXAMPLE BELOW CREATES A GRID WITH 4 RANDOM STENCILS PLACED
# IN THE SQUARE [-1,1]^2

p1 = 5 # number of random stencils
domain1 = ComputationalDomain(N = N, a = -1, b = 1, c = -1, d = 1)
coordinates1 = np.random.randint(N, size=(p1, 2)) # Generating random coordinates for boundary conditions
coordinates1 = [tuple(sub) for sub in coordinates1]
domain1.Gamma(stencils = coordinates1)

solver1 = EikonalSolver(domain = domain1)
solver1.SweepUntilConvergence(epsilon = 1e-7, verbose = True)
np.save('random5stencils_grid.npy',solver1.grids_after_sweeps[-1])


# SECOND EXAMPLE WITH HIGHER NUMBER OF RANDOMLY PUT STENCILS FOR GAMMA
p2 = 30 # number of random stencils
domain2 = ComputationalDomain(N = N, a = -1, b = 1, c = -1, d = 1)
coordinates2 = np.random.randint(N, size=(p2, 2)) # Generating random coordinates for boundary conditions
coordinates2 = [tuple(sub) for sub in coordinates2]
domain2.Gamma(stencils = coordinates2)

solver2 = EikonalSolver(domain = domain2)
solver2.SweepUntilConvergence(epsilon = 1e-7, verbose = True)
np.save('random30stencils_grid.npy',solver2.grids_after_sweeps[-1])

