import numpy as np
import matplotlib.pyplot as plt
from computational_domain import ComputationalDomain


def ReLU(x):
    return np.max([x, 0])


#### CLASS DEFINING THE SOLVER
class EikonalSolver:
    def __init__(self, domain, F):
        self.domain = domain
        self.grids_after_sweeps = []
        self.grids_after_sweeps.append(self.domain.grid)
        self.F = F #Velocity on the domain

    def update_point_with_F(self, current_grid, new_grid, i, j):
        f_ij = 1/self.F(self.domain.a + j*self.domain.h, self.domain.d - i*self.domain.h) #converting from numpy coordinates to cartesian
        N = self.domain.N
        u_old = current_grid[i, j]

        if (i, j) in self.domain.frozen:  # meaning we're encountering a point in the BC Gamma
            return

        u_xmin = np.min([new_grid[i, j - 1] if j > 0 else np.inf,
                         new_grid[i, j + 1] if j < N - 1 else np.inf])
        u_ymin = np.min([new_grid[i - 1, j] if i > 0 else np.inf,
                         new_grid[i + 1, j] if i < N - 1 else np.inf])

        # Conditions on updating u_old

        if ReLU(u_old - u_xmin) == 0 and ReLU(u_old - u_ymin) != 0:
            u_bar = f_ij*self.domain.h + u_ymin

        elif ReLU(u_old - u_ymin) == 0 and ReLU(u_old - u_xmin) != 0:
            u_bar = f_ij*self.domain.h + u_xmin

        elif ReLU(u_old - u_xmin) == 0 and ReLU(u_old - u_ymin) == 0:
            return  # No update needed

        else:
            a = u_xmin
            b = u_ymin

            if np.abs(b - a) >= f_ij*self.domain.h:
                u_bar = np.min([a, b]) + f_ij*self.domain.h

            else:
                u_bar = (a + b + np.sqrt(2*(f_ij*self.domain.h)**2 - (a - b)**2)) / 2

        # UPDATE OF THE POINT
        new_grid[i, j] = np.min([u_old, u_bar])

    def Sweep1(self):
        """Sweep going from bottom left corner of the domain to top-right corner"""
        N = self.domain.N
        current_grid = self.grids_after_sweeps[-1].copy()
        new_grid = self.grids_after_sweeps[-1].copy()
        for i in reversed(range(N)):
            for j in range(N):
                self.update_point_with_F(current_grid, new_grid, i, j)
        self.grids_after_sweeps.append(new_grid) # The grid after the 1st Sweep

    def Sweep2(self):
        """Sweep going from bottom right corner of the domain to the top-left corner"""
        N = self.domain.N
        current_grid = self.grids_after_sweeps[-1].copy()
        new_grid = self.grids_after_sweeps[-1].copy()

        for i in reversed(range(N)):
            for j in reversed(range(N)):
                self.update_point_with_F(current_grid, new_grid, i, j)

        self.grids_after_sweeps.append(new_grid)

    def Sweep3(self):
        """Sweep going from bottom right corner of the domain to the top-left corner"""
        N = self.domain.N
        current_grid = self.grids_after_sweeps[-1].copy()
        new_grid = self.grids_after_sweeps[-1].copy()
        for i in range(N):
            for j in reversed(range(N)):
                self.update_point_with_F(current_grid, new_grid, i, j)

        self.grids_after_sweeps.append(new_grid)

    def Sweep4(self):
        """Sweep going from bottom right corner of the domain to the top-left corner"""
        N = self.domain.N
        current_grid = self.grids_after_sweeps[-1].copy()
        new_grid = self.grids_after_sweeps[-1].copy()
        for i in range(N):
            for j in range(N):
                self.update_point_with_F(current_grid, new_grid, i, j)
        self.grids_after_sweeps.append(new_grid)

    def BatchSweeps(self, k=1):
        for i in range(k):
            self.Sweep1()
            self.Sweep2()
            self.Sweep3()
            self.Sweep4()

    def SweepUntilConvergence(self, epsilon = 1e-3, verbose = False):
        """Runs sweeps one by one, checking for convergence within an epsilon
        threshold at each step, using the L-infinity norm"""
        k = 0
        while True:
            prev = self.grids_after_sweeps[-1].copy()

            self.Sweep1()
            k += 1
            if np.max(np.abs(self.grids_after_sweeps[-1] - prev)) < epsilon:
                break
            prev = self.grids_after_sweeps[-1].copy()

            self.Sweep2()
            k += 1
            if np.max(np.abs(self.grids_after_sweeps[-1] - prev)) < epsilon:
                break
            prev = self.grids_after_sweeps[-1].copy()

            self.Sweep3()
            k += 1
            if np.max(np.abs(self.grids_after_sweeps[-1] - prev)) < epsilon:
                break
            prev = self.grids_after_sweeps[-1].copy()

            self.Sweep4()
            k += 1
            if np.max(np.abs(self.grids_after_sweeps[-1] - prev)) < epsilon:
                break
            prev = self.grids_after_sweeps[-1].copy()

        if verbose:
            print(f'Convergence after {k} iterations')









