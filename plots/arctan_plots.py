import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from sklearn.linear_model import LinearRegression #for fitting errors

errors_atan = np.load('../results/general_f/2D_arctan_errors.npy')
index = np.load('../results/general_f/N_index_errors_arctan.npy')

# CONVERGENCE PLOT FOR ERROR VS GRID SIZE N
plt.style.use(['science', 'grid']) #from scienceplots for better plots

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


# CONVERGENCE PLOT FOR ERROR VS STEP SIZE H
#We fit the linear model in the error
model = LinearRegression()
model2 = LinearRegression()
model3 = LinearRegression()
X = np.array([(1/(N-1)) for N in index]).reshape(-1,1)
model.fit(np.sqrt(X), errors_atan)
model2.fit(np.abs(X*np.log(X)),errors_atan)
model3.fit(X**2,errors_atan)
A = float(model2.coef_)
B = float(model2.intercept_)
print('R^2 for sqrt h : ',model.score(np.sqrt(X), errors_atan))
print('R^2 for hlogh : ',model2.score(np.abs(X*np.log(X)), errors_atan))
print('R^2 for h^2 : ',model3.score(np.abs(X**2), errors_atan))
x = np.linspace(np.min(X), np.max(X),150)
plt.style.use(['science', 'grid'])
plt.figure(figsize=(7.8, 4.8))
plt.plot([(1/(N-1)) for N in index][::-1], np.flip(errors_atan),  marker='x', label=r'$L^{\infty}$ Error')
plt.plot(x, A*np.abs(x*np.log(x)) + B, label = fr'${round(A,4)} \cdot |h \log h| + {round(B,4)}$')
plt.xlabel(r'Step size $h$')
plt.ylabel("Max Error")
plt.yscale("log")
plt.title(r'Convergence of Eikonal Solver with $F(x,y)=1+0.5(x^{2}+y^{2})$ on $[-1,1]^2$, $\epsilon = 10^{-6}$')
plt.legend()
plt.tight_layout()
plt.savefig('convergence2d_atan_step_size.png', dpi=600, bbox_inches='tight')
plt.show(block=False)


# 3D SURFACE OF THE NUMERICAL SOLUTION

x = np.linspace(-2, 2, 801)
y = np.linspace(-2, 2, 801)

X, Y = np.meshgrid(x, y)
Z = np.load('../results/general_f/grid_atan_N801.npy')

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none')

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$u(x, y)$')
ax.set_title('Numerical Solution for $F(x,y) = 1+0.5(x^2+y^2)$ on $[-2,2]^{2}$, $N=801$', fontsize = 18)
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.tight_layout()
plt.savefig('solution_3d_surface_arctan.png', dpi=600)
plt.show(block=True)


# CONTOURS OF THE SOLUTION

plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels = 15, cmap = 'jet')
plt.colorbar(contour)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Contours of the numerical solution over the square $[-2,2]^{2}$', fontsize = 18, pad  = 20)
plt.tight_layout()
plt.savefig('contour_plot_arctan.png', dpi=600, bbox_inches='tight')
plt.show()