"""
This code is used to visualize stored results from one time of running the method
"""
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

plt.close("all")

#Set this variable the data path to the EEG dataset
data_path = '../EEG/'

#Set this variable the path for the result folder
result_path = '../Code_Signal/Result/'

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
#Reconstrution
Dz = output['Dz']
#list of objective function's values after each iteration
list_obj_val =output['list_obj_val']
#list of objective function's values after each d update
list_obj_val_filter = output['list_obj_val_filter']
#list of objective function's values after each z update
list_obj_val_z = output['list_obj_val_z']

#Only take the main part of the reconstruction, not the added border part by the kernel
Dz = Dz[:,100:6100]
#Only take the main part of the codes, not the added border part by the kernel
z = z[:,:,100:6100]
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
                    
#Draw the first singla and its first 10 codes corresponding to the first 10 kernels
colors = ['r','g','b']
nrow = 11
id_signal = 0
fig4 = plt.figure(114)
fig4.add_subplot(nrow, 1, 1)
plt.plot(b[0])
plt.axis('off')
for i in range(nrow-1):
    fig4.add_subplot(nrow, 1, i+2)
    plt.plot(z[id_signal, i, :], color=colors[np.mod(i,3)])
    plt.axis('off')
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.15, hspace=0.05)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
fig4.canvas.set_window_title('codes_' + str(id_signal))