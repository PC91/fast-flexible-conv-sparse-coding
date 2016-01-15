import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

A = np.load('big_images_fast.npz')

d=A['d']
z=A['z']
Dz=A['Dz']

nrow = 10
ncol = 10

fig1 = plt.figure(1)

for i in range(1):
    for j in range(10):
        fig1.add_subplot(nrow, ncol, i * ncol + j + 1)
        plt.imshow(Dz[i * ncol + j], cmap = cm.Greys_r, vmin = np.min(Dz), vmax = np.max(Dz))
        plt.axis('off')
        
plt.savefig('reconstruction.png', bbox_inches='tight')

fig2 = plt.figure(2)

for i in range(nrow):
    for j in range(ncol):
        fig2.add_subplot(nrow, ncol, i * ncol + j + 1)
        plt.imshow(d[i * ncol + j], cmap = cm.Greys_r, vmin = np.min(d), vmax = np.max(d))
        plt.axis('off')

plt.savefig('filters.png', bbox_inches='tight')
        
fig3 = plt.figure(3)

for i in range(nrow):
    for j in range(ncol):
        fig3.add_subplot(nrow, ncol, i * ncol + j + 1)
        plt.imshow(z[0][i * ncol + j], cmap = cm.Greys_r, vmin = np.min(z), vmax = np.max(z))
        plt.axis('off')

plt.savefig('coding.png', bbox_inches='tight')