import matplotlib.pyplot as plt
import numpy as np

grid = np.load('cosxsiny.npy')

plt.imshow(grid)
plt.colorbar()
plt.title('numerical solution')
plt.show()

## true solution

x = np.linspace(-2,2,301)
y = np.linspace(-2,2,301)
X,Y = np.meshgrid(x,y)

Z = np.cos(X)*np.sin(Y)

plt.imshow(Z, origin='lower')
plt.colorbar()
plt.title('true solution')
plt.show()