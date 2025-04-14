import numpy as np

def bilinear_gradient(u, x, y, domain):
    h = domain.h
    x_min, x_max = domain.a, domain.b
    y_min, y_max = domain.c, domain.d
    N = domain.N

    j = int((x - x_min) / h)
    i = int((y_max - y) / h)

    if not (0 <= i < N-1 and 0 <= j < N-1):
        return np.array([0.0, 0.0])  # or raise exception

    x1 = x_min + j*h
    y1 = y_max - i*h

    tx = (x - x1) / h
    ty = (y1 - y) / h

    u11 = u[i+1, j]
    u21 = u[i+1, j+1]
    u12 = u[i, j]
    u22 = u[i, j+1]

    ux = ((1 - ty)*(u21 - u11) + ty*(u22 - u12))/h
    uy = ((1 - tx)*(u12 - u11) + tx*(u22 - u21))/h

    return np.array([ux, uy])


test  = np.load('../results/distance/2D_distance_grid.npy')
from computational_domain import ComputationalDomain
n = 256
unit_square = ComputationalDomain(N = n+1, a = -1, b = 1, c = -1, d = 1)

print(bilinear_gradient(test, 0.5, 0.5, unit_square))
