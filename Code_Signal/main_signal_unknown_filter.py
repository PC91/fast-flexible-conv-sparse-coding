"""
This code is used to find a kernel and the corresponding codes for signal data
"""
import numpy as np
from scipy.io import loadmat
import CSC_signal as CSC
import time

start = time.clock()

#Set this variable the data path to the EEG dataset
#ata_path = '/home/chau/Dropbox/PRIM/EEG/'
data_path = '../EEG/'

#Set this variable the path for the result folder
result_path = '../Code_Signal/Result/'

#Load data to be run
mat = loadmat(data_path + 'eeg_data.mat')
eeg_signal = mat['eeg_signal']
eeg_label = mat['eeg_label']

#Extract the wanted subset of data  
eeg_signal = np.concatenate((eeg_signal[0:1,:], eeg_signal[20:21,:],\
                             eeg_signal[40:41,:], eeg_signal[60:61,:],\
                             eeg_signal[80:81,:]), axis=0)

n_signal = eeg_signal.shape[0] #the number of signals
n_sample = eeg_signal.shape[1] #dimension of each signal

#Normalize the data to be run
b = eeg_signal
b = ((b.T - np.mean(b, axis = 1)) / np.std(b, axis = 1)).T

#Define the parameters
#128 kernels with size of 201
size_kernel = [128, 201]

#Optim options
max_it = 30 #the number of iterations
tol = np.float64(1e-3) #the stop threshold for the algorithm 

#RUN THE ALGORITHM
[d, z, Dz, list_obj_val, list_obj_val_filter, list_obj_val_z, reconstr_err] = \
        CSC.learn_conv_sparse_coder(b, size_kernel, max_it, tol)
  
#Save into an external struct .npz the kernel, the codes, the reconstruction
#and list of objective function's values     
np.savez(result_path + 'result_eeg.npz',\
         d=d, z=z, Dz=Dz, list_obj_val=list_obj_val,\
         list_obj_val_filter=list_obj_val_filter, list_obj_val_z=list_obj_val_z)

end = time.clock()

print end - start