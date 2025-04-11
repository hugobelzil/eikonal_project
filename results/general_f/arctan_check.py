import matplotlib.pyplot as plt
import numpy as np

temp = np.load("arctan_grid.npy")
plt.imshow(temp)
plt.title('numerical solution')
plt.colorbar()
plt.show()

x = np.linspace(-1,1, 401)
y = np.linspace(-1,1, 401)
X, Y = np.meshgrid(x,y)
Z = np.sqrt(2)*np.arctan(np.sqrt(X**2 + Y**2)/np.sqrt(2))

plt.imshow(Z)
plt.title("true solution")
plt.colorbar()
plt.show()