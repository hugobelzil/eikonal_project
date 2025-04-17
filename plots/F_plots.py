import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from sklearn.linear_model import LinearRegression #for fitting error

errors_atan = np.load('../results/general_f/2D_arctan_errors.npy')
index = np.load('../results/general_f/N_index_errors_arctan.npy')

# Convergence plot for error vs. grid size N
plt.style.use(['science', 'grid'])
plt.figure(figsize=(7.8, 4.8))
plt.plot(index, errors_atan,  marker='x', label=r'$L^{\infty}$ Error')
plt.xlabel("Grid size (N)")
plt.ylabel("Max Error")
plt.yscale("log")
plt.title(r'Convergence of Eikonal Solver $F(x,y)=1+0.5(x^{2}+y^{2})$ function on $[-1,1]^2$, $\epsilon = 10^{-6}$')
plt.legend()
plt.tight_layout()
plt.savefig('convergence_atan_grid_size.png', dpi=600, bbox_inches='tight')
plt.show(block=True)  # Keep the window open


# Convergence plot for error vs. step size h
#We fit the linear model in the error
model = LinearRegression()
X = np.array([(2/(N-1)) for N in index]).reshape(-1,1)
model.fit(np.abs(X*np.log(X)), errors_atan)
A = float(model.coef_)
B = float(model.intercept_)

x = np.linspace(np.min(X), np.max(X),150)
plt.style.use(['science', 'grid'])
plt.figure(figsize=(7.8, 4.8))
plt.plot([(2/(N-1)) for N in index][::-1], np.flip(errors_atan),  marker='x', label=r'$L^{\infty}$ Error')
plt.plot(x, A*np.abs(x*np.log(x)) + B, label = fr'${round(A,4)} \cdot |h \log h| + {round(B,4)}$')
plt.xlabel(r'Step size $h$')
plt.ylabel("Max Error")
plt.yscale("log")
plt.title(r'Convergence of Eikonal Solver with $F(x,y)=1+0.5(x^{2}+y^{2})$ on $[-1,1]^2$, $\epsilon = 10^{-6}$')
plt.legend()
plt.tight_layout()
plt.savefig('convergence2d_atan_step_size.png', dpi=600, bbox_inches='tight')
plt.show(block=True)


# CONTOUR OF THE NUMERICAL SOLUTION
