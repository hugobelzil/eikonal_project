import numpy as np
from interpolation.interpolator import bilinear_gradient

def ODE_backtracer(x0, dt, domain, u_grid, max_steps=100000, tol=1e-8):
    """ Returns shortest path (obtained via gradient descent)"""
    path = [np.array(x0)]
    x = x0[0]
    y = x0[1]
    for _ in range(max_steps):
        grad = bilinear_gradient(u_grid, x, y, domain)
        norm_grad = np.linalg.norm(grad)

        if norm_grad < tol:
            break
        # Euler step along -∇u direction (normalized)
        x -= dt*grad[0]
        y -= dt*grad[1]

        # Stop if outside domain
        if not (domain.a <= x <= domain.b and domain.c <= y <= domain.d):
            break

        path.append(np.array([x, y]))
    return np.array(path)