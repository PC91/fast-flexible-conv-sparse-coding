"""
This code is used to visualize the reconstruction and codes w.r.t. beta
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

A = np.load('reconstr_err_wrt_beta.npz')

list_reconstr_err_wrt_beta=A['list_reconstr_err_wrt_beta']

#Draw the reconstruction errors
fig0 = plt.figure(0)
plt.plot(np.arange(0.5, 5.1, 0.5), list_reconstr_err_wrt_beta[0:10])
plt.xlabel('Beta')
plt.ylabel('Reconstruction error')
plt.savefig('reconstr_err_wrt_beta.png')


nrow = 10
ncol = 11

#Draw the 10 codes and the reconstructed image of the 10 original images
fig1 = plt.figure(1)
cbar_ax_code = fig1.add_axes([0, 0.15, 0.05, 0.7])
cbar_ax_recon = fig1.add_axes([0.92, 0.15, 0.05, 0.7])
for i in range(nrow):
    I = np.load('medium_10_5_5_beta_' + str((i + 1) / 2.0) + '.npz')
    z = I['z']
    Dz = I['Dz']
    for j in range(ncol):    
        fig1.add_subplot(nrow, ncol, i * ncol + j + 1)
        if (j == ncol - 1):
            #reconstructed image
            im_recon = plt.imshow(Dz[0], cmap = cm.Greys_r)
        else:
            #sparse codes
            im_code = plt.imshow(abs(z[0][j]), cmap = cm.Greys_r, vmin = np.min(abs(z)), vmax = np.max(abs(z)))
        plt.axis('off')
plt.subplots_adjust(left=0.08, bottom=0.0, right=0.9, top=1.0, wspace=0.15, hspace=0.05)
fig1.colorbar(im_code, cax=cbar_ax_code)
fig1.colorbar(im_recon, cax=cbar_ax_recon)
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.savefig('coding_wrt_beta.png')