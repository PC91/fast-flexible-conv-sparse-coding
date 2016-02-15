"""
This code is used to find a kernel and the corresponding codes for image data
"""
import numpy as np
from PIL import Image
import CSC_image as CSC
import os
import time

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

start = time.clock()

#Set this variable the data path to the image dataset 
#path = '/home/chau/Dropbox/PRIM/small_images/'
path = '/Users/huydinh/Dropbox/PRIM/medium_images/'

#Get the images
fileList_gen = listdir_nohidden(path)
fileList = list(fileList_gen)
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

#Define the parameters
#10 kernels with size of 5 x 5
size_kernel = [10, 5, 5]

#Optim options
max_it = 20 #number of iterations
tol = np.float64(1e-3) #stop threshold

#List of testing beta
list_reconstr_err_wrt_beta = np.zeros(10)

#Run the algorithm with beta from 0.5 to 5.0 by the increase of 0.5
for beta in np.arange(0.5, 5.1, 0.5):
    [d, z, Dz, list_obj_val, list_obj_val_filter, list_obj_val_z, reconstr_err] = \
        CSC.learn_conv_sparse_coder(b, size_kernel, max_it, tol, beta)
    list_reconstr_err_wrt_beta[beta * 2 - 1] = reconstr_err
           
    np.savez('medium_10_5_5_beta_' + str(beta) + '.npz', d=d, z=z, Dz=Dz,
                          list_obj_val=list_obj_val,
                          list_obj_val_filter=list_obj_val_filter,
                          list_obj_val_z=list_obj_val_z)

#Save the list of reconstruction errors with respect to beta                      
np.savez('reconstr_err_wrt_beta.npz', list_reconstr_err_wrt_beta=list_reconstr_err_wrt_beta)
end = time.clock()

print end - start