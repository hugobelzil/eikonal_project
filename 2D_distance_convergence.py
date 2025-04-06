import numpy as np
import matplotlib.pyplot as plt
from eikonal_distance_2D import EikonalSolver
from computational_domain import ComputationalDomain
import scienceplots

# Activate SciencePlots style
plt.style.use(['science', 'grid'])

errors = []

for N in [(20*i+1) for i in range(1,5)]:
    print("Running convergence analysis. N = ", N)
    unit_square = ComputationalDomain(N = N, a = -1, b = 1, c = -1, d = 1)
    unit_square.Gamma([(int((N-1)/2), int((N-1)/2))])

    solver = EikonalSolver(unit_square)
    solver.SweepUntilConvergence(epsilon = 1e-4)
    numerical_solution = solver.grids_after_sweeps[-1]

    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    true_distance = np.sqrt(X**2 + Y**2)

    max_error = np.max(np.abs(true_distance - numerical_solution))
    errors.append(max_error)
    print("N = ", N, " max error : ", max_error)



plt.figure(figsize=(6.5, 4))
plt.plot([(25*i +1) for i in range(1,5)], errors, marker='x', label=r'$L^\infty$ Error')
plt.xlabel("Grid size (N)")
plt.ylabel("Max Error")
plt.yscale("log")
plt.title("Convergence of Eikonal Solver")
plt.legend()
plt.tight_layout()
plt.show(block=True)  # Keep the window open

