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
        f_ij = 1/F(self.domain.a + j*self.domain.h, self.domain.d - i*self.domain.h) #converting from numpy coordinates to cartesian
        N = self.domain.N
        u_old = current_grid[i, j]

        if u_old == 1:
            return # no update needed for points in Gamma

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


def F(x,y):
    return (1+0.5*(x**2+y**2))

test = ComputationalDomain(N=601, a=-1, b=1, c=-1, d=1)
test.Gamma([(300, 300)])
print(test.grid)
print(test.h)

solver = EikonalSolver(test, F)
solver.BatchSweeps(k=5)
print("AFTER SWEEP 1 : \n", solver.grids_after_sweeps[1])
print("AFTER SWEEP 2 : \n", solver.grids_after_sweeps[2])
print("AFTER SWEEP 3 : \n", solver.grids_after_sweeps[7])
print("AFTER SWEEP 4 : \n", solver.grids_after_sweeps[8])
#print(len(solver.grids_after_sweeps))
#plt.imshow(solver.grids_after_sweeps[4])
#print(solver.grids_after_sweeps[4][6,0])
#plt.show()

print("Exact value in the corner : ", np.sqrt(2)*np.arctan(1))
print("check : grid after sweep at gamma : ", solver.grids_after_sweeps[-1][150,150])
print("value in the corner after 3 sweeps : ", solver.grids_after_sweeps[-1][0,0])








