import numpy as np
import matplotlib.pyplot as plt

a = 5
x = np.linspace(-1, 1, 1000)*a
y = np.linspace(-1, 1, 1000)*a

X, Y = np.meshgrid(x, y)

Z = X**2 + Y**2
fig = plt.figure(figsize=(10, 10))
plt.imshow(Z, cmap='Blues_r', alpha=0.8)
plt.contour(Z, colors='black', alpha=0.3)
plt.scatter(500, 500, color='red')
plt.clim(0,25)
plt.axis('off')

plt.savefig('../../plots/harmonic_potential.png', bbox_inches='tight')
plt.close()