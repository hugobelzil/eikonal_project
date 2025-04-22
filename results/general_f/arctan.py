import numpy as np
from solvers.eikonal_general_F_2D import EikonalSolver
from computational_domain import ComputationalDomain


# The velocity with gives solution u(x,y) = sqrt2*atan(sqrt(x**2+y**2)/srqt2)
def F(x, y):
    return 1 + 0.5*(x**2 + y**2)

A = 36
epsilon = 1e-6
errors = []

for N in [(20*i + 1) for i in range(10, A)]:
    print("Running convergence analysis. N = ", N)
    unit_square = ComputationalDomain(N = N, a = -1, b = 1, c = -1, d = 1)
    unit_square.Gamma([(int((N-1)/2), int((N-1)/2))])

    solver = EikonalSolver(unit_square, F = F)
    solver.SweepUntilConvergence(epsilon = epsilon)
    numerical_solution = solver.grids_after_sweeps[-1]

    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    true_solution = np.sqrt(2)*np.arctan(np.sqrt(X**2 + Y**2)/np.sqrt(2))


    max_error = np.max(np.abs(true_solution - numerical_solution))
    errors.append(max_error)
    print("N = ", N, " max error : ", max_error)

print("Convergence analysis finished. Saving results.")
np.save('2D_arctan_errors.npy', errors)
np.save('N_index_errors_arctan.npy', [(20*i + 1) for i in range(10,A)])

# EXAMPLE ON LARGER GRID (FOR PLOTTING PURPOSES)
domain2 = ComputationalDomain(N = 801, a = -2, b = 2, c = -2, d = 2)
domain2.Gamma([(400,400)])
solver = EikonalSolver(domain2, F = F)
print('Solving on larger grid...')
solver.SweepUntilConvergence(epsilon = epsilon)
np.save('grid_atan_N801.npy',solver.grids_after_sweeps[-1])
print('DONE')