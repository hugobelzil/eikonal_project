import numpy as np
from solvers.eikonal_distance_2D import EikonalSolver
from solvers.computational_domain import ComputationalDomain

# This script aims at comparing the order of accuracy of the fast sweeping method
# for distance functions when performing 4 sweeps versus 5 sweeps, for non-single
# boundary conditions. As highlighted in the paper of H. Zhao, we should expect
# the numerical solution to be O(hlogh) after 4 sweeps and O(h) after 5 sweeps

errors = []
A = 4 #16

for N in [(250*i + 1) for i in range(2, A)]:
    domain = ComputationalDomain(N = N, a = -1, b = 1, c = -1, d = 1)
    domain.Gamma([(int((N-1)/2)), int((N-1)/2)])