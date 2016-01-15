"""
- This code implements a solver for the paper "Fast and Flexible Convolutional Sparse Coding".
- The goal of this solver is to find the common filters, the codes for each image in the dataset
and a reconstruction for the dataset.
- The common filters and the codes for each image is denoted as d and z respectively
- We denote the step to solve the filter is d-step and the codes z-step
"""
import numpy as np
#from numpy import linalg
from scipy import linalg
import filehelp as fh           #DEBUG
from scipy.fftpack import fft2, ifft2

real_type = 'float64'
imaginary_type = 'complex128'
    
def learn_conv_sparse_coder(b, size_kernel, lambda_residual, lambda_prior, max_it, tol):
    """
    Main function to solve the convolutional sparse coding
    Parameters for this function
    - b               : the image dataset with size (num_images, height, width)
    - size_kernel     : the size of each kernel (num_kernels, height, width)
    - lambda_residual :
    - lambda_prior    :
    - max_it          : the maximum iterations of the outer loop
    - tol             : the minimal difference in filters and codes after each iteration to continue
    - init            : the initial parameters, may be
                        + d (filters): to find the output codes by a predefined filter set
                        + z (code): to find the output filters by a predefined code set
                        
    Important variables used in the code:
    - u_D, u_Z        : pair of proximal values for d-step and z-step
    - d_D, d_Z        : pair of Lagrange multipliers in the ADMM algo for d-step and z-step
    - v_D, v_Z        : pair of initial value pairs (Zd, d) for d-step and (Dz, z) for z-step
    """
   
    psf_s = size_kernel[1]
    k     = size_kernel[0]
    n     = b.shape[0]
                
    psf_radius = int(np.floor(psf_s/2))
    
    size_x = [n, b.shape[1] + 2*psf_radius, b.shape[2] + 2*psf_radius]
    size_z = [n, k, size_x[1], size_x[2]]
    #size_k = [k, 2*psf_radius + 1, 2*psf_radius + 1]
    size_k_full = [k, size_x[1], size_x[2]]
       
    #M is MtM, Mtb is Mtx, the matrix M is zero-padded in 2*psf_radius rows and cols
    #M = padarray(ones(size(b)), [psf_radius, psf_radius, 0], 0, 'both');
    #Mtb = padarray(b, [psf_radius, psf_radius, 0], 0, 'both');
    
    M   = np.pad(np.ones(b.shape, dtype = real_type),
                 ((0,0), (psf_radius, psf_radius), (psf_radius, psf_radius)),
                 mode='constant', constant_values=0)
    Mtb = np.pad(b, ((0, 0), (psf_radius, psf_radius), (psf_radius, psf_radius)),\
                 mode='constant', constant_values=0)
               
    
    #Penalty parameters
    lambdas = [lambda_residual, lambda_prior]
    gamma_heuristic = 60 * lambda_prior * 1/np.amax(b)
    gammas_D = [gamma_heuristic / 5000, gamma_heuristic] #[gamma_heuristic / 2000, gamma_heuristic];    
    gammas_Z = [gamma_heuristic / 500, gamma_heuristic] #[gamma_heuristic / 2, gamma_heuristic];
    rho = gammas_D[1]/gammas_D[0]
    
    #Initialize variables for K
    varsize_D = [size_x, size_k_full]
    xi_D     = [np.zeros(varsize_D[0], dtype = real_type),
                np.zeros(varsize_D[1], dtype = real_type)]
                
    xi_D_hat = [np.zeros(varsize_D[0], dtype = imaginary_type),
                np.zeros(varsize_D[1], dtype = imaginary_type)]
    
    u_D = [np.zeros(varsize_D[0], dtype = real_type),
           np.zeros(varsize_D[1], dtype = real_type)]
    #LAGRANGE MULTIPLIERS TO SOLVE WITH D-STEP       
    d_D = [np.zeros(varsize_D[0], dtype = real_type),
           np.zeros(varsize_D[1], dtype = real_type)]
           
    v_D = [np.zeros(varsize_D[0], dtype = real_type),
           np.zeros(varsize_D[1], dtype = real_type)]
    
    #Initial main variables, for each iteration on d or z
    #The filters
    d = np.random.normal(size=size_kernel)
    d = np.pad(d, ((0, 0),
                   (0, size_x[1] - size_kernel[1]),
                   (0, size_x[2] - size_kernel[2])),
               mode='constant', constant_values=0)
    d = np.roll(d, -int(psf_radius), axis=1)
    d = np.roll(d, -int(psf_radius), axis=2)
    #fh.read_file('/home/chau/Dropbox/PRIM/d.txt', d, 3)
    d_hat = fft2(d)
    
    #Initialize variables for Z
    varsize_Z = [size_x, size_z]
    xi_Z = [np.zeros(varsize_Z[0], dtype = real_type),
            np.zeros(varsize_Z[1], dtype = real_type)]
            
    xi_Z_hat = [np.zeros(varsize_Z[0], dtype = imaginary_type),
                np.zeros(varsize_Z[1], dtype = imaginary_type)]
    
    u_Z = [np.zeros(varsize_Z[0], dtype = real_type),
           np.zeros(varsize_Z[1], dtype = real_type)]
    #LAGRANGE MULTIPLIERS TO SOLVE WITH Z-STEP              
    d_Z = [np.zeros(varsize_Z[0], dtype = real_type),
           np.zeros(varsize_Z[1], dtype = real_type)]
           
    v_Z = [np.zeros(varsize_Z[0], dtype = real_type),
           np.zeros(varsize_Z[1], dtype = real_type)]
    
    ##The code
    z = np.random.normal(size=size_z)
    #fh.read_file('/home/chau/Dropbox/PRIM/z.txt', z, 4)
    z_hat = fft2(z)
    
    #Initial objective function (usually very large)
    obj_val = obj_func(z_hat, d_hat, b,
                       lambda_residual, lambda_prior,
                       psf_radius, size_z, size_x)
       
    #Back-and-forth local iteration for d and z
    max_it_d = 10
    max_it_z = 10
    
    obj_val_filter = obj_val
    obj_val_z = obj_val
    
    #Start the main algorithm
    for i in range(max_it):
        #Update kernels
        #Recompute what is necessary for kernel convterm later
        [zhat_mat, zhat_inv_mat] = precompute_H_hat_D(z_hat, size_z, rho)
        
        obj_val_min = min(obj_val_filter, obj_val_z)
        
        d_old = d
        d_hat_old = d_hat
        
        #UPDATE FILTERS
        for i_d in range(max_it_d):
           
            #Compute v = Z * d
            d_hat_dot_z_hat = np.multiply(d_hat, z_hat)
            v_D[0] = np.real(ifft2(np.sum(d_hat_dot_z_hat, axis=1).reshape(size_x)))
            v_D[1] = d

            #Line 3: Compute proximal updates
            u = v_D[0] - d_D[0]
            theta = lambdas[0]/gammas_D[0]
            u_D[0] = np.divide((Mtb + 1.0/theta * u), (M + 1.0/theta * np.ones(size_x)))
            
            u = v_D[1] - d_D[1]
            u_D[1] = KernelConstraintProj(u, size_k_full, psf_radius)
            
            #Line 4: Update Langrange multipliers
            d_D[0] = d_D[0] + (u_D[0] - v_D[0])
            d_D[1] = d_D[1] + (u_D[1] - v_D[1])
            
            #Compute new xi and transform to fft
            xi_D[0] = u_D[0] + d_D[0]
            xi_D[1] = u_D[1] + d_D[1]
            xi_D_hat[0] = fft2(xi_D[0])
            xi_D_hat[1] = fft2(xi_D[1])
            #Line 2: Solve convolutional inverse
            #d = ( sum_j(gamma_j * H_j'* H_j) )^(-1) * ( sum_j(gamma_j * H_j'* xi_j) )
            d_hat = solve_conv_term_D(zhat_mat, zhat_inv_mat, xi_D_hat, rho, size_z)
            d = np.real(ifft2(d_hat))                      
            
            if (i_d == max_it_d - 1):
                obj_val = obj_func(z_hat, d_hat, b,
                                   lambda_residual, lambda_prior, 
                                   psf_radius, size_z, size_x)
                
                print('--> Obj %3.3f'% obj_val)
                   
        obj_val_filter = obj_val
        
        #Debug progress
        d_diff = d - d_old
        d_comp = d

        obj_val = obj_func(z_hat, d_hat, b,
                           lambda_residual, lambda_prior, 
                           psf_radius, size_z, size_x)
        print('Iter D %d, Obj %3.3f, Diff %5.5f'%
              (i, obj_val, linalg.norm(d_diff) / linalg.norm(d_comp)))
               
               
        #UPDATE SPARSITY CODES
        
        #Recompute what is necessary for convterm later
        [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(d_hat, size_x)        
        dhatT_flat = np.ma.conjugate(dhat_flat.T)
        
        z_old = z
        z_hat_old = z_hat
               
        for i_z in range(max_it_z):
            
            #Compute v = D * z
            d_hat_dot_z_hat = np.multiply(d_hat, z_hat)
            v_Z[0] = np.real(ifft2(np.sum(d_hat_dot_z_hat, axis=1).reshape(size_x)))
            v_Z[1] = z

            #Compute proximal updates
            u = v_Z[0] - d_Z[0]
            theta = lambdas[0]/gammas_Z[0]
            u_Z[0] = np.divide((Mtb + 1.0/theta * u ), (M + 1.0/theta * np.ones(size_x)))
            
            u = v_Z[1] - d_Z[1]
            theta = lambdas[1]/gammas_Z[1] * np.ones(u.shape)                        
            u_Z[1] = np.multiply(np.maximum(0, 1 - np.divide(theta, np.abs(u))), u)

            #Update running errors
            d_Z[0] = d_Z[0] + (u_Z[0] - v_Z[0])
            d_Z[1] = d_Z[1] + (u_Z[1] - v_Z[1])
            
            #Compute new xi and transform to fft
            xi_Z[0] = u_Z[0] + d_Z[0]
            xi_Z[1] = u_Z[1] + d_Z[1]

            xi_Z_hat[0] = fft2(xi_Z[0])
            xi_Z_hat[1] = fft2(xi_Z[1])
            
            #Solve convolutional inverse
            # z = ( sum_j(gamma_j * H_j'* H_j) )^(-1) * ( sum_j(gamma_j * H_j'* xi_j) )
            z_hat = solve_conv_term_Z(dhatT_flat, dhatTdhat_flat, xi_Z_hat, gammas_Z, size_z)
            z = np.real(ifft2(z_hat))
            
            #z_matlab = np.zeros(z.shape)
            #fh.read_file('/home/chau/Dropbox/PRIM/z.txt', z_matlab, 4)
            
            if (i_z == max_it_z - 1):
                obj_val = obj_func(z_hat, d_hat, b,
                                   lambda_residual, lambda_prior, 
                                   psf_radius, size_z, size_x)
                
                print('--> Obj %3.3f'% obj_val)
                    
        obj_val_z = obj_val
        
        if (obj_val_min <= obj_val_filter and obj_val_min <= obj_val_z):
            z_hat = z_hat_old
            z = np.real(ifft2(z_hat))
            
            d_hat = d_hat_old
            d = np.real(ifft2(d_hat))
            
            obj_val = obj_func(z_hat, d_hat, b,
                               lambda_residual, lambda_prior, 
                               psf_radius, size_z, size_x)
            break
        
        #Debug progress
        z_diff = z - z_old
        z_comp = z
        print('Iter Z %d, Obj %3.3f, Diff %5.5f'%
              (i, obj_val, linalg.norm(z_diff) / linalg.norm(z_comp)))
        
        #Termination
        if (linalg.norm(z_diff) / linalg.norm(z_comp) < tol and
            linalg.norm(d_diff) / linalg.norm(d_comp) < tol):
            break
    
    #Final estimate
    z_res = z
    
    d_res = d
    d_res = np.roll(d_res, psf_radius, axis=1)
    d_res = np.roll(d_res, psf_radius, axis=2)
    d_res = d_res[:, 0:psf_radius*2+1, 0:psf_radius*2+1] 
    
    fft_dot = np.multiply(d_hat, z_hat)
    Dz = np.real(ifft2(np.sum(fft_dot, axis=1).reshape(size_x)))
    
    obj_val = obj_func(z_hat, d_hat, b,
                       lambda_residual, lambda_prior, 
                       psf_radius, size_z, size_x)
    print('Final objective function %f'% (obj_val))
    return [d_res, z_res, Dz]
        

def KernelConstraintProj(u, size_k_full, psf_radius):
    
    #size_x = [n, b.shape[0] + 2*psf_radius, b.shape[1] + 2*psf_radius]
    #size_z = [n, k, size_x[1], size_x[2]]
    
    #Get support
    u_proj = u
    u_proj = np.roll(u_proj, psf_radius, axis=1)
    u_proj = np.roll(u_proj, psf_radius, axis=2)
    u_proj = u_proj[:, 0:psf_radius*2+1, 0:psf_radius*2+1]
    
    #Normalize
    u_sum = np.sum(np.sum(np.power(u_proj, 2), axis=1), axis=1)
    u_norm = np.tile(u_sum, [u_proj.shape[1], u_proj.shape[2], 1]).transpose(2,0,1)
    u_proj[u_norm >= 1] = u_proj[u_norm >= 1] / np.sqrt(u_norm[u_norm >= 1])
    
    #Now shift back and pad again   
    u_proj = np.pad(u_proj, ((0,0),
                             (0, size_k_full[1] - (2*psf_radius+1)),
                             (0, size_k_full[2] - (2*psf_radius+1))),
                    mode='constant', constant_values=0)
       
    u_proj = np.roll(u_proj, -psf_radius, axis=1)
    u_proj = np.roll(u_proj, -psf_radius, axis=2)    

    return u_proj

def precompute_H_hat_D(z_hat, size_z, rho):
    
    #Computes the spectra for the inversion of all H_i
    
    #Size
    n = size_z[0]
    k = size_z[1]
    
    #Precompute spectra for H    
    #zhat_mat = np.ndarray.transpose(z_hat.transpose(0,1,3,2).reshape(n, k, -1), [2, 0, 1])
    zhat_mat = np.transpose(z_hat.transpose(0,1,3,2).reshape(n, k, -1), [2, 0, 1])    
    
    #Precompute the inverse matrices for each frequency    
    zhat_inv_mat = np.zeros((zhat_mat.shape[0], k, k), dtype=imaginary_type)
    
    inv_rho_z_hat_z_hat_t = np.zeros((zhat_mat.shape[0], n, n), dtype=imaginary_type)
    
    z_hat_mat_t = np.transpose(np.ma.conjugate(zhat_mat), [0, 2, 1])
    
    z_hat_z_hat_t = np.einsum('knm,kmj->knj',zhat_mat, z_hat_mat_t)#.real
    
    #NOT SURE IF THIS PART COULD BE ACCELERATED OR NOT    (I)
    for i in range(zhat_mat.shape[0]):
        #inv_rho_z_hat_z_hat_t[i] = np.linalg.pinv(rho * np.eye(n) + z_hat_z_hat_t[i])
        this_zz = z_hat_z_hat_t[i]
        this_zz.flat[::n + 1] += rho
        inv_rho_z_hat_z_hat_t[i] = linalg.pinv(this_zz)
                         
    zhat_inv_mat = 1.0/rho * (np.eye(k) - 
                              np.einsum('knm,kmj->knj',                              
                                        np.einsum('knm,kmj->knj',
                                                  z_hat_mat_t,
                                                  inv_rho_z_hat_z_hat_t),
                                        zhat_mat))
    
    print ('Done precomputing for D')
    
    #UNVECTORIZED CODE OF (I)
#    for i in range(zhat_mat.shape[0]):
#        
#        z_hat_mat_t = np.ma.conjugate(zhat_mat[i,:]).T
#        zhat_inv_mat[i,:] = \
#                1.0/rho * np.eye(k) - \
#                1.0/rho * np.dot(np.dot(z_hat_mat_t,
#                                        np.linalg.pinv(rho * np.eye(n) +
#                                                       np.dot(zhat_mat[i,:], z_hat_mat_t))),
#                                 zhat_mat[i,:])
                                 
    return [zhat_mat, zhat_inv_mat]

def precompute_H_hat_Z(dhat, size_x):
    
    #Computes the spectra for the inversion of all H_i

    #Precompute the dot products for each frequency
    dhat_flat = dhat.transpose(0,2,1).reshape((-1, size_x[1] * size_x[2])).T
    dhatTdhat_flat = np.sum(np.multiply(np.ma.conjugate(dhat_flat), dhat_flat), axis=1)
    
    return [dhat_flat, dhatTdhat_flat]

def solve_conv_term_D(zhat_mat, zhat_inv_mat, xi_hat, rho, size_z):

    #Solves sum_j gamma_i/2 * || H_j d - xi_j ||_2^2
    #In our case: 1/2|| Zd - xi_1 ||_2^2 + rho * 1/2 * || d - xi_2||
    #with rho = gammas[1]/gammas[0]
    
    #Size
    n = size_z[0]
    k = size_z[1]
    sy = size_z[2]
    sx = size_z[3]

    #Reshape to array per freq////.>uency
    xi_hat_0_flat = np.expand_dims(np.reshape(xi_hat[0].transpose(0,2,1),
                                              (n, sx * sy)).T,
                                   axis=2)
    xi_hat_1_flat = np.expand_dims(np.reshape(xi_hat[1].transpose(0,2,1),
                                              (k, sx * sy)).T,
                                   axis=2)
    
    #Invert
    x = np.zeros((zhat_mat.shape[0], k), dtype=imaginary_type)
    z_hat_mat_t = np.ma.conjugate(zhat_mat.transpose(0,2,1))
    x = np.einsum("ijk, ikl -> ijl", zhat_inv_mat,
                                     np.einsum("ijk, ikl -> ijl",
                                               z_hat_mat_t,
                                               xi_hat_0_flat) +
                                     rho * xi_hat_1_flat) \
                  .reshape(sx * sy, k)
        
    #Reshape to get back the new Dhat
    d_hat = np.reshape(x.T, (k, sx, sy)).transpose(0,2,1)
    
    return d_hat
    
def solve_conv_term_Z(dhatT, dhatTdhat, xi_hat, gammas, size_z ):

    #Solves sum_j gamma_i/2 * || H_j z - xi_j ||_2^2
    #In our case: 1/2|| Dz - xi_1 ||_2^2 + rho * 1/2 * || z - xi_2||
    #with rho = gamma(2)/gamma(1)
    
    #Size
    n = size_z[0]
    k = size_z[1]
    sy = size_z[2]
    sx = size_z[3]
    
    #Rho
    rho = gammas[1]/gammas[0]
    
    #Compute b       
    xi_hat_0_rep = xi_hat[0].transpose([0,2,1]).reshape(n, 1, sy*sx)
    xi_hat_1_rep = xi_hat[1].transpose([0,1,3,2]).reshape(n, k, sy*sx)
    
    b = np.multiply(dhatT, xi_hat_0_rep) + rho * xi_hat_1_rep
    
    #Invert
    scInverse = np.divide(np.ones((1, sx*sy)), rho*np.ones((1, sx*sy)) + dhatTdhat.T)
    
    dhatT_dot_b = np.multiply(np.ma.conjugate(dhatT), b)
    dhatTb_rep = np.expand_dims(np.sum(dhatT_dot_b, axis=1), axis=1)
    x = 1.0/rho * (b - np.multiply(np.multiply(scInverse, dhatT), dhatTb_rep))
    
    #Final transpose gives z_hat
    temp_size = np.zeros(4)
    temp_size[0] = size_z[0]
    temp_size[1] = size_z[1]
    temp_size[3] = size_z[2]
    temp_size[2] = size_z[3]
    z_hat = x.reshape(temp_size).transpose(0,1,3,2)
    
    return z_hat
      
def obj_func(z_hat, d_hat, b, lambda_residual, lambdas, \
                      psf_radius, size_z, size_x):
    
    #Dataterm
    d_hat_dot_z_hat = np.multiply(d_hat, z_hat)
    Dz = np.real(ifft2(np.sum(d_hat_dot_z_hat, axis=1).reshape(size_x)))
    
    f_z = lambda_residual * 1.0/2.0 * \
          np.power(linalg.norm(np.reshape(Dz[:, psf_radius:(Dz.shape[1] - psf_radius), \
                                                psf_radius:(Dz.shape[2] - psf_radius)] - b, \
                                          -1, 1)), 2)
    #Regularizer
    z = ifft2(z_hat)
    g_z = lambdas * np.sum(np.abs(z))
    
    f_val = f_z + g_z
    return f_val