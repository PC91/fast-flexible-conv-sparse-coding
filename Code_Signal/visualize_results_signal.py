"""
This code is used to visualize stored results from one time of running the method
"""
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

plt.close("all")

#Set this variable the data path to the EEG dataset
data_path = '/home/chau/Dropbox/PRIM/EEG/'
#data_path = '/Users/huydinh/Dropbox/PRIM/EEG/'

#Set this variable the path for the result folder
result_path = '/home/chau/Dropbox/PRIM/Code_Signal/Result/'
#result_path = '/Users/huydinh/Dropbox/PRIM/Code_Signal/Result/'

#Load data to be run
mat = loadmat(data_path + 'eeg_data.mat')
eeg_signal = mat['eeg_signal']
eeg_label = mat['eeg_label']

#Load the available result from previous times of running
output = np.load(result_path + 'result_eeg_50_128_201.npz')
#Kernel
d = output['d']
#Sparse codes
z = output['z']
#Reconstruction
Dz = output['Dz']
#list of objective function's values after each iteration
list_obj_val =output['list_obj_val']
#list of objective function's values after each d update
list_obj_val_filter = output['list_obj_val_filter']
#list of objective function's values after each z update
list_obj_val_z = output['list_obj_val_z']

#Only take the main part of the reconstruction, not the added part by the kernel
Dz = Dz[:,100:6100]
#Make a signal become 0 if it is very close to 0
z[np.max(z,axis=2)<0.005] = 0

#Number of signals
n_signal = eeg_signal.shape[0]
#Dimention of each signal
n_sample = eeg_signal.shape[1]
#Number of kernels
n_filter = d.shape[0]

#Normalize the data
b = eeg_signal
b = ((b.T - np.mean(b, axis = 1)) / np.std(b, axis = 1)).T

#Extract the corresponding original data for the stored result
b = np.concatenate((b[0:10,:], b[20:30,:],\
                    b[40:50,:], b[60:70,:],\
                    b[80:90,:]), axis=0)

#Draw the objective function's values over iterations                   
fig0 = plt.figure(0)
plt.plot(list_obj_val_z[5:30])
plt.xticks(np.arange(0, 25, 2), np.arange(6, 31, 2))
plt.xlabel('Current iteration')
plt.ylabel('Objective function')
plt.savefig(result_path+'objective_func.png')


#Draw 5 sample original signals
fig1 = plt.figure(1)
for i in range(5):
    fig1.add_subplot(10, 1, i+1)
    plt.plot(b[i * 10, :])
    plt.axis('off')
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, hspace=0.05)
figManager = plt.get_current_fig_manager()             
fig1.canvas.set_window_title('original signal')
plt.savefig(result_path+'original_signal.png')

#Draw 5 corresponding reconstructed signals                    
fig2 = plt.figure(2)
for i in range(5):
    fig2.add_subplot(10, 1, i+1)
    plt.plot(Dz[i * 10, :])
    plt.axis('off')
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, hspace=0.05)
figManager = plt.get_current_fig_manager()
fig2.canvas.set_window_title('reconstruction')
plt.savefig(result_path+'reconstruction.png')


#Draw 128 kernels
nrow = 16
ncol = 8
fig3 = plt.figure(3)
for i in range(nrow):
    for j in range(ncol):
        if (i * ncol + j < n_filter):
            fig3.add_subplot(nrow, ncol, i * ncol + j + 1)
            plt.plot(d[i*ncol+j, :])
            plt.axis('off')
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.15, hspace=0.05)
figManager = plt.get_current_fig_manager()        
fig3.canvas.set_window_title('filters')
plt.savefig(result_path+'filters.png')
  
#Draw 128 codes corresponding to 128 kernels of signal id_signal, herer the 0th signal       
nrow = 16
ncol = 8
id_signal = 0
fig4 = plt.figure(114)
for i in range(nrow):
    for j in range(ncol):
        if (i * ncol + j < n_filter):
            fig4.add_subplot(nrow, ncol, i * ncol + j + 1)
            plt.plot(z[id_signal, i * ncol + j, :])
            plt.axis('off')
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.15, hspace=0.05)
figManager = plt.get_current_fig_manager()           
fig4.canvas.set_window_title('codes_' + str(id_signal))
plt.savefig(result_path+'codes_' + str(id_signal) + '.png')

#Draw 128 codes corresponding to 128 kernels of signal id_signal, herer the 10th signal
nrow = 16
ncol = 8
id_signal = 10
fig5 = plt.figure(115)
for i in range(nrow):
    for j in range(ncol):
        if (i * ncol + j < n_filter):
            fig5.add_subplot(nrow, ncol, i * ncol + j + 1)
            plt.plot(z[id_signal, i * ncol + j, :])
            plt.axis('off')
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.15, hspace=0.05)
figManager = plt.get_current_fig_manager()          
fig5.canvas.set_window_title('codes_' + str(id_signal))
plt.savefig(result_path+'codes_' + str(id_signal) + '.png')

#Draw 128 codes corresponding to 128 kernels of signal id_signal, herer the 20th signal
nrow = 16
ncol = 8
id_signal = 20
fig6 = plt.figure(116)
for i in range(nrow):
    for j in range(ncol):
        if (i * ncol + j < n_filter):
            fig6.add_subplot(nrow, ncol, i * ncol + j + 1)
            plt.plot(z[id_signal, i * ncol + j, :])
            plt.axis('off')

plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.15, hspace=0.05)
figManager = plt.get_current_fig_manager()          
fig6.canvas.set_window_title('codes_' + str(id_signal))
plt.savefig(result_path+'codes_' + str(id_signal) + '.png')

#Draw 128 codes corresponding to 128 kernels of signal id_signal, herer the 30th signal
nrow = 16
ncol = 8
id_signal = 30
fig7 = plt.figure(117)
for i in range(nrow):
    for j in range(ncol):
        if (i * ncol + j < n_filter):
            fig7.add_subplot(nrow, ncol, i * ncol + j + 1)
            plt.plot(z[id_signal, i * ncol + j, :])
            plt.axis('off')
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.15, hspace=0.05)
figManager = plt.get_current_fig_manager()           
fig7.canvas.set_window_title('codes_' + str(id_signal))
plt.savefig(result_path+'codes_' + str(id_signal) + '.png')

#Draw 128 codes corresponding to 128 kernels of signal id_signal, herer the 40th signal
nrow = 16
ncol = 8
id_signal = 40
fig8 = plt.figure(118)
for i in range(nrow):
    for j in range(ncol):
        if (i * ncol + j < n_filter):
            fig8.add_subplot(nrow, ncol, i * ncol + j + 1)
            plt.plot(z[id_signal, i * ncol + j, :])
            plt.axis('off')

plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.15, hspace=0.05)
figManager = plt.get_current_fig_manager()          
fig8.canvas.set_window_title('codes_' + str(id_signal))
plt.savefig(result_path+'codes_' + str(id_signal) + '.png')