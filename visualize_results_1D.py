import numpy as np
import matplotlib.pyplot as plt
import os

path = '/Users/huydinh/Dropbox/PRIM/Signal code/'


A = np.load('result4.npz')

d=A['d']
z=A['z']
Dz=A['Dz']

n_signal = 14

n_sample = 14980

n_filter = 10

b = np.zeros((n_signal, n_sample), dtype=np.float64)

data = np.loadtxt('EEG_Eye_State.arff.txt', comments = '@', delimiter = ',')

b = data[:, 0:n_signal].T

b = b[:, np.max(b, axis = 0) < 8000]

b = ((b.T - np.mean(b, axis = 1)) / np.std(b, axis = 1)).T


nrow = 4
ncol = 4

fig1 = plt.figure(0)

for i in range(nrow):
    for j in range(ncol):
        if (i * ncol + j < n_signal):
            fig1.add_subplot(nrow, ncol, i * ncol + j + 1)
            plt.plot(b[i * ncol + j, :])
            plt.axis('off')
    
plt.savefig('original_singal.png')
        
fig1 = plt.figure(1)

for i in range(nrow):
    for j in range(ncol):
        if (i * ncol + j < n_signal):
            fig1.add_subplot(nrow, ncol, i * ncol + j + 1)
            plt.plot(Dz[i * ncol + j, :])
            plt.axis('off')
        #plt.imshow(Dz[i * ncol + j], cmap = cm.Greys_r)
        #plt.axis('off')
#plt.tight_layout()        
plt.savefig('reconstruction.png')

fig2 = plt.figure(2)

for i in range(3):
    for j in range(4):
        if (i * ncol + j < n_filter):
            fig2.add_subplot(nrow, ncol, i * ncol + j + 1)
            plt.plot(d[i * ncol + j, :])
            plt.axis('off')

plt.savefig('filters.png')
        
fig3 = plt.figure(3)

for i in range(3):
    for j in range(4):
        if (i * ncol + j < n_filter):
            fig3.add_subplot(nrow, ncol, i * ncol + j + 1)
            plt.plot(z[0, i * ncol + j, :])
            plt.axis('off')

plt.savefig('coding.png')#, bbox_inches='tight')

fig4 = plt.figure(4)

plt.plot(b[0, :])

plt.savefig('signal_0.png')