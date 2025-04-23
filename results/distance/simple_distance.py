import numpy as np
from solvers.eikonal_distance_2D import EikonalSolver
from computational_domain import ComputationalDomain

# BUILDING THE UNIT SQUARE
n = 256
unit_square = ComputationalDomain(N = n+1, a = -1, b = 1, c = -1, d = 1)
unit_square.Gamma([(int(n/2), int(n/2))])

# CREATE AN INSTANCE OF THE SOLVER AND SOLVING
solver = EikonalSolver(unit_square)
solver.SweepUntilConvergence(epsilon = 1e-6, verbose = True)

# SAVING THE GRID IN A NUMPY OBJECT
np.save('2D_distance_grid.npy', solver.grids_after_sweeps[-1])
print("PDE solved on unit square grid \n")

# CONVERGENCE ANALYSIS:
errors = []
epsilon = 1e-4
A = 16 # Max value for N is 20*A + 1 = 461 here
print("Now running convergence analysis...")

for N in [(500*i + 1) for i in range(2, A)]:
    print("Running convergence analysis. N = ", N)
    unit_square = ComputationalDomain(N = N, a = -1, b = 1, c = -1, d = 1)
    unit_square.Gamma([(int((N-1)/2), int((N-1)/2))])

    solver = EikonalSolver(unit_square)
    #solver.SweepUntilConvergence(epsilon = epsilon, verbose = True)
    solver.BatchSweeps(k = 1)
    numerical_solution = solver.grids_after_sweeps[-1]

    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    true_distance = np.sqrt(X**2 + Y**2)


    max_error = np.max(np.abs(true_distance - numerical_solution))
    errors.append(max_error)
    print("N = ", N, " max error : ", max_error)

print("Convergence analysis finished. Saving results.")
np.save('2D_distance_errors.npy', errors)
np.save('N_index_errors.npy', [(500*i + 1) for i in range(2,A)])