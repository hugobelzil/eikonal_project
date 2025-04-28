import numpy as np

### Class allowing to build the computational domains
### on which all methods implemented in this repository run
class ComputationalDomain:
    def __init__(self, N, a, b, c, d):
        """Constructor : takes as input the number of points N to make
        an NxN Numpy array, representing the [a,b]x[c,d] square in R^2"""
        self.grid =  np.round(100*np.ones((N, N)),10) #np.round(np.sqrt(2)*np.max([b-a,d-c])*np.ones((N,N)),4)
        self.N = int(N)
        self.h = (b-a)/(N-1)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.frozen = [] # Stores the boundary points

    def Gamma(self, stencils):
        """Adds the boundary conditions on the grid."""
        if type(stencils) != list:
            raise Exception("List of coordinates for Gamma points (side-conditions) must be of type List")

        for point in stencils:
            self.grid[point]= int(0)
            self.frozen.append(point)