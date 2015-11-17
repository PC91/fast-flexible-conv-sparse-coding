import numpy as np
from PIL import Image
import convolutionalsparsecoding as CSC
# import filehelp as fh
import os
# import itertools

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
 
#path = '/home/chau/Dropbox/PRIM/small_images/'
# path = '/Users/huydinh/Dropbox/PRIM/single_image/'
# path = './single_image/'
path = './small_images/'

fileList_gen = listdir_nohidden(path)

#filename = next(itertools.islice(fileList, 3, 4))
fileList = list(fileList_gen)
#print filename
#print (sum(1 for _ in fileList))
#print fileList[0]
sample_image = Image.open(path + fileList[0], 'r')

b = np.zeros((len(fileList), sample_image.size[1], sample_image.size[0]), dtype=np.float64)

#Normalize each image to zero mean and unit variance
idx = 0
for infile in fileList:
    image = Image.open(path + infile, 'r').convert('L')
    b[idx,:] = np.asarray(image.getdata(),dtype=np.float64).\
               reshape((image.size[1], image.size[0]))
    b[idx,:] = b[idx,:] / 255;
    b[idx,:] = (b[idx,:] - np.mean(b[idx,:])) / np.std(b[idx,:])
    
    idx += 1

#fh.read_file('/home/chau/Dropbox/PRIM/b.txt', b, 3)

#Define the parameters
size_kernel = [10, 5, 5]
lambda_residual = np.float64(1.0)
lambda_prior = np.float64(1.0)

#Optim options
max_it = 20
tol = np.float64(1e-3)

[d, z, Dz] = CSC.learn_conv_sparse_coder(b, size_kernel, lambda_residual, lambda_prior, max_it, tol)
#print d.size, z.size, Dz.size
np.savez('single_image.npz', d=d, z=z, Dz=Dz)