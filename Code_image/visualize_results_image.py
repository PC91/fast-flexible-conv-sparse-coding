"""
This code is used to visualize stored results from one time of running the method
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Load the result struct
A = np.load('medium_10_5_5.npz')
#Kernel
d=A['d']
#Sparse codes
z=A['z']
#Reconstructed images
Dz=A['Dz']
#List of objective function's values over iterations
list_obj_val_z = A['list_obj_val_z']

#Draw the objective function's values over iterations 
fig0 = plt.figure(0)
plt.plot(list_obj_val_z[1:10])
plt.xticks(np.arange(0, 9), np.arange(2, 11))
plt.xlabel('Current iteration')
plt.ylabel('Objective function')
plt.savefig('objective_func.png')

nrow = 1
ncol = 10

#Draw 10 reconstructed images
fig1 = plt.figure(1)
for i in range(1):
    for j in range(10):
        fig1.add_subplot(nrow, ncol, i * ncol + j + 1)
        plt.imshow(Dz[i * ncol + j], cmap = cm.Greys_r, vmin = np.min(Dz), vmax = np.max(Dz))
        plt.axis('off')       
plt.savefig('reconstruction.png', bbox_inches='tight')

#Draw 10 kernels
fig2 = plt.figure(2)
for i in range(nrow):
    for j in range(ncol):
        fig2.add_subplot(nrow, ncol, i * ncol + j + 1)
        plt.imshow(d[i * ncol + j], cmap = cm.Greys_r, vmin = np.min(d), vmax = np.max(d))
        plt.axis('off')
plt.colorbar()
plt.savefig('filters_chau.png', bbox_inches='tight')
   
#Draw 10 codes corresponding to 10 kernels of the first image     
fig3 = plt.figure(3)
for i in range(nrow):
    for j in range(ncol):
        fig3.add_subplot(nrow, ncol, i * ncol + j + 1)
        plt.imshow(abs(z[0][i * ncol + j]), cmap = cm.Greys_r, vmin = np.min(abs(z[0])), vmax = np.max(abs(z[0])))
        plt.axis('off')
plt.savefig('coding.png', bbox_inches='tight')