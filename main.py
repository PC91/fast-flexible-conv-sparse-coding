import numpy as np
from PIL import Image
import convolutionalsparsecoding as CSC
import filehelp as fh
import os
 
path = '/home/chau/Dropbox/PRIM/small_images/'
fileList = os.listdir(path)

sample_image = Image.open(path + fileList[0], 'r')

b = np.zeros((len(fileList), sample_image.size[1], sample_image.size[0]), dtype=np.float64)

idx = 0
for infile in fileList:
    image = Image.open(path + infile, 'r').convert('L')
    b[idx,:] = np.asarray(image.getdata(),dtype=np.float64).\
               reshape((image.size[1], image.size[0]))
    b[idx,:] = b[idx,:] / 255;
    b[idx,:] = (b[idx,:] - np.mean(b[idx,:])) / np.std(b[idx,:])
    
    idx += 1

#Define the parameters
size_kernel = [10, 5, 5]
lambda_residual = np.float64(1.0)
lambda_prior = np.float64(1.0)

#Optim options
max_it = 20
tol = np.float64(1e-3)

[d, z, Dz] = CSC.learn_conv_sparse_coder(b, size_kernel, lambda_residual, lambda_prior, max_it, tol)