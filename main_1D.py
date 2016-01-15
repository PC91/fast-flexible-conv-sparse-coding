import numpy as np
from PIL import Image
import convolutionalsparsecoding_1D as CSC
import filehelp as fh
import os
import time
import itertools


start = time.clock()
 
#path = '/home/chau/Dropbox/PRIM/small_images/'
path = '/Users/huydinh/Dropbox/PRIM/Signal code/'



n_signal = 14

n_sample = 14980

b = np.zeros((n_signal, n_sample), dtype=np.float64)

data = np.loadtxt('EEG_Eye_State.arff.txt', comments = '@', delimiter = ',')



b = data[:, 0:n_signal].T

b = b[:, np.max(b, axis = 0) < 8000]

b = ((b.T - np.mean(b, axis = 1)) / np.std(b, axis = 1)).T

#fh.read_file('/home/chau/Dropbox/PRIM/b.txt', b, 3)

#Define the parameters
size_kernel = [10, 11]
lambda_residual = np.float64(1.0)
lambda_prior = np.float64(1.0)

#Optim options
max_it = 50
tol = np.float64(1e-3)

[d, z, Dz] = CSC.learn_conv_sparse_coder(b, size_kernel, lambda_residual, lambda_prior, max_it, tol)
#print d.size, z.size, Dz.size
np.savez('result4.npz', d=d, z=z, Dz=Dz)

end = time.clock()

print end - start