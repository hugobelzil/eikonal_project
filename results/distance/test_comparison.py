import numpy as np
from solvers.eikonal_distance_2D import EikonalSolver
from solvers.computational_domain import ComputationalDomain
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

errors_epsilon = []
errors_4sweeps = []

for N in [(20*i + 1) for i in range(2,16)]:
    print('Running analysis for N = ', N)
    domain = ComputationalDomain(N = N, a = -1, b = 1, c = -1, d = 1)
    domain.Gamma([(int((N-1)/2), int((N-1)/2))])
    solv1 = EikonalSolver(domain = domain) #4 SWEEPS
    solv2 = EikonalSolver(domain = domain) #CONVERGENCE UNDER EPSILON
    solv1.BatchSweeps()
    solv2.BatchSweeps(k = 2)

    solution_4sweeps = solv1.grids_after_sweeps[-1]
    solution_epsilon = solv2.grids_after_sweeps[-1]

    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    true_distance = np.sqrt(X ** 2 + Y ** 2)
    true_l2_norm = np.sqrt(8/3)

    l2_norm_4sweeps = np.sum(solution_4sweeps**2)*((domain.h)**2)
    l2_norm_4sweeps = np.sqrt(l2_norm_4sweeps)

    l2_norm_epsilon = np.sum(solution_epsilon ** 2)*((domain.h)**2)
    l2_norm_epsilon = np.sqrt(l2_norm_epsilon)

    max_error_4sweeps = np.max(np.abs(true_distance - solution_4sweeps))
    max_error_epsilon = np.max(np.abs(true_distance - solution_epsilon))

    l2_error_4sweeps = np.abs(l2_norm_4sweeps - true_l2_norm)
    l2_error_epsilon = np.abs(l2_norm_epsilon - true_l2_norm)
    #print('error 4 sweeps : ', max_error_4sweeps)
    #print('error epsilon : ', max_error_epsilon)
    print('True L2 NORM : ', true_l2_norm)
    print('L2 norm for epsilon : ', l2_norm_epsilon)
    print('L2 norm for 4 sweeps : ', l2_norm_4sweeps)
    #errors_epsilon.append(max_error_epsilon)
    #errors_4sweeps.append(max_error_4sweeps)
    errors_epsilon.append(l2_error_epsilon)
    errors_4sweeps.append(l2_error_4sweeps)



index = [(20*i + 1) for i in range(2,16)]
model = LinearRegression()
X = np.array([(2/(N-1)) for N in index]).reshape(-1,1)
model.fit(np.abs(X*np.log(X)), errors_epsilon)
A = float(model.coef_)
B = float(model.intercept_)
x = np.linspace(np.min(X),np.max(X), 150)

plt.figure(figsize=(7.8, 4.8))
plt.plot([(2/(N-1)) for N in index][::-1], np.flip(errors_epsilon),  marker='x', label=r'$L^{\infty}$ Error')
plt.plot(x, A*np.abs(x*np.log(x)) + B, label = fr'${round(A,4)} \cdot |h \log h| + {round(B,4)}$')
plt.xlabel(r'Step size $h$')
plt.ylabel("Max Error")
plt.yscale("log")
plt.title(r'Convergence of Eikonal Solver for distance function on $[-1,1]^2$, $\epsilon = 10^{-6}$')
plt.legend()
plt.tight_layout()
plt.show(block=True)

plt.figure(figsize=(7.8, 4.8))
plt.plot([(2/(N-1)) for N in index][::-1], np.flip(errors_4sweeps),  marker='x', label=r'$L^{\infty}$ Error')
#plt.plot(x, A*np.abs(x*np.log(x)) + B, label = fr'${round(A,4)} \cdot |h \log h| + {round(B,4)}$')
plt.xlabel(r'Step size $h$')
plt.ylabel("Max Error")
plt.yscale("log")
plt.title(r'Convergence of Eikonal Solver for distance function on $[-1,1]^2$, 4 sweeps only')
plt.legend()
plt.tight_layout()
plt.show(block=True)