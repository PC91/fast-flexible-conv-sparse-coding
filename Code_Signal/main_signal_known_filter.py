"""
This code is used to find codes for signal data when the filter is already known
"""
import numpy as np
from scipy.io import loadmat
import CSC_signal as CSC
import time

start = time.clock()

#Set this variable the data path to the EEG dataset
data_path = '../EEG/'

#Set this variable the path for the result folder
result_path = '../Code_Signal/Result/'

#Load the available result from previous times of running
output = np.load(result_path + 'result_eeg_50_128_201.npz')

#d is the known filter
d = output['d']
#list of objective function's values after each iteration
list_obj_val = output['list_obj_val']
#list of objective function's values after each d update
list_obj_val_filter = output['list_obj_val_filter']
#list of objective function's values after each z update
list_obj_val_z = output['list_obj_val_z']


#Load data to be run
mat = loadmat(data_path + 'eeg_data.mat')
eeg_signal = mat['eeg_signal']
eeg_label = mat['eeg_label']

#Extract the wanted subset of data 
eeg_signal = eeg_signal[99,:]
eeg_signal = eeg_signal[None, :]

n_signal = 1 #the number of signals
n_sample = eeg_signal.shape[1] #dimension of each signal

#Normalize the data to be run
b = eeg_signal
b = ((b.T - np.mean(b, axis = 1)) / np.std(b, axis = 1)).T

#Define the parameters
#Size of d, i.e. 128 kernels with size of 201
size_kernel = [128, 201]

#Optim options
max_it = 30 #the number of iterations
tol = np.float64(1e-3) #the stop threshold for the algorithm 

#RUN THE ALGORITHM
[temp, z, Dz, list_obj_val, list_obj_val_filter, list_obj_val_z, reconstr_err] = \
        CSC.learn_conv_sparse_coder(b, size_kernel, max_it, tol,\
                                    known_d=d)
        
end = time.clock()

print end - start