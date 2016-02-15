"""
- This code implements a solver for the paper "Fast and Flexible Convolutional Sparse Coding" on signal data.
- The goal of this solver is to find the common filters, the codes for each signal series in the dataset
and a reconstruction for the dataset.
- The common filters (or kernels) and the codes for each image is denoted as d and z respectively
- We denote the step to solve the filter as d-step and the codes z-step
"""
import numpy as np
from scipy import linalg
from scipy.fftpack import fft, ifft

real_type = 'float64'
imaginary_type = 'complex128'
    
def learn_conv_sparse_coder(b, size_kernel, max_it, tol,
                            known_d=None,
                            beta=np.float64(1.0)):
    """
    Main function to solve the convolutional sparse coding
    Parameters for this function
    - b               : the signal dataset with size (num_signals, length)
    - size_kernel     : the size of each kernel (num_kernels, length)
    - beta            : the trade-off between sparsity and reconstruction error (as mentioned in the paper)
    - max_it          : the maximum iterations of the outer loop
    - tol             : the minimal difference in filters and codes after each iteration to continue
    - known_d         : the predefined filters (if possible)
                        
    Important variables used in the code:
    - u_D, u_Z        : pair of proximal values for d-step and z-step
    - d_D, d_Z        : pair of Lagrange multipliers in the ADMM algo for d-step and z-step
    - v_D, v_Z        : pair of initial value pairs (Zd, d) for d-step and (Dz, z) for z-step
    """
   
    k     = size_kernel[0]
    n     = b.shape[0]

    psf_radius = int(np.floor(size_kernel[1]/2))
    
    size_x = [n, b.shape[1] + 2*psf_radius]
    size_z = [n, k, size_x[1]]
    size_k_full = [k, size_x[1]]
       
    #M is MtM, Mtb is Mtx, the matrix M is zero-padded in 2*psf_radius rows and cols   
    M   = np.pad(np.ones(b.shape, dtype = real_type),
                 ((0,0), (psf_radius, psf_radius)),
                 mode='constant', constant_values=0)
    Mtb = np.pad(b, ((0, 0), (psf_radius, psf_radius)),\
                 mode='constant', constant_values=0)
               
    
    """Penalty parameters, including the calculation of augmented Lagrange multipliers"""
    lambda_residual=np.float64(1.0)
    lambda_prior=np.float64(1.0)    
    lambdas = [lambda_residual, lambda_prior]
    gamma_heuristic = 60 * lambda_prior * 1/np.amax(b)
    gammas_D = [gamma_heuristic / 5000, gamma_heuristic]
    gammas_Z = [gamma_heuristic / 500, gamma_heuristic]
    rho = gammas_D[1]/gammas_D[0]
       
    """Initialize variables for the d-step"""
    if known_d is None:
        varsize_D = [size_x, size_k_full]
        xi_D     = [np.zeros(varsize_D[0], dtype = real_type),
                    np.zeros(varsize_D[1], dtype = real_type)]
                    
        xi_D_hat = [np.zeros(varsize_D[0], dtype = imaginary_type),
                    np.zeros(varsize_D[1], dtype = imaginary_type)]
        
        u_D = [np.zeros(varsize_D[0], dtype = real_type),
               np.zeros(varsize_D[1], dtype = real_type)]
        
        #Lagrange multipliers
        d_D = [np.zeros(varsize_D[0], dtype = real_type),
               np.zeros(varsize_D[1], dtype = real_type)]
               
        v_D = [np.zeros(varsize_D[0], dtype = real_type),
               np.zeros(varsize_D[1], dtype = real_type)]
               
        d = np.random.normal(size=size_kernel)
    else:
        d = known_d
    
    #Initial the filters and its fft after being rolled to fit the frequency
    d = np.pad(d, ((0, 0),
                   (0, size_x[1] - size_kernel[1])),
               mode='constant', constant_values=0)
    d = np.roll(d, -int(psf_radius), axis=1)
    d_hat = fft(d)
    
    """Initialize variables for the z-step"""
    varsize_Z = [size_x, size_z]
    xi_Z = [np.zeros(varsize_Z[0], dtype = real_type),
            np.zeros(varsize_Z[1], dtype = real_type)]
            
    xi_Z_hat = [np.zeros(varsize_Z[0], dtype = imaginary_type),
                np.zeros(varsize_Z[1], dtype = imaginary_type)]
    
    u_Z = [np.zeros(varsize_Z[0], dtype = real_type),
           np.zeros(varsize_Z[1], dtype = real_type)]

    #Lagrange multipliers
    d_Z = [np.zeros(varsize_Z[0], dtype = real_type),
           np.zeros(varsize_Z[1], dtype = real_type)]
           
    v_Z = [np.zeros(varsize_Z[0], dtype = real_type),
           np.zeros(varsize_Z[1], dtype = real_type)]
    
    #Initial the codes and its fft
    z = np.random.normal(size=size_z)
    z_hat = fft(z)
    
    
    """Initial objective function (usually very large)"""
    obj_val = obj_func(z_hat, d_hat, b,
                       lambda_residual, lambda_prior,
                       psf_radius, size_z, size_x)
       
    #Back-and-forth local iteration for d and z
    if known_d is None:
        max_it_d = 10
    else:
        max_it_d = 0
        
    max_it_z = 10
    
    obj_val_filter = obj_val
    obj_val_z = obj_val
    
    list_obj_val = np.zeros(max_it)
    list_obj_val_filter = np.zeros(max_it)
    list_obj_val_z = np.zeros(max_it)
    
    """Start the main algorithm"""
    for i in range(max_it):
        
        """D-STEP"""
        if known_d is None:
            #Precompute what is necessary for later            
            [zhat_mat, zhat_inv_mat] = precompute_D_step(z_hat, size_z, rho)
            d_old = d
        
            for i_d in range(max_it_d):
               
               #Compute v = [Zd, d]
                d_hat_dot_z_hat = np.multiply(d_hat, z_hat)
                v_D[0] = np.real(ifft(np.sum(d_hat_dot_z_hat, axis=1).reshape(size_x)))
                v_D[1] = d
    
                #Compute proximal updates
                u = v_D[0] - d_D[0]
                theta = lambdas[0]/gammas_D[0]
                u_D[0] = np.divide((Mtb + 1.0/theta * u), (M + 1.0/theta * np.ones(size_x)))
                
                u = v_D[1] - d_D[1]
                u_D[1] = KernelConstraintProj(u, size_k_full, psf_radius)
                
                #Update Langrange multipliers
                d_D[0] = d_D[0] + (u_D[0] - v_D[0])
                d_D[1] = d_D[1] + (u_D[1] - v_D[1])
                
                #Compute new xi=u+d and transform to fft
                xi_D[0] = u_D[0] + d_D[0]
                xi_D[1] = u_D[1] + d_D[1]
                xi_D_hat[0] = fft(xi_D[0])
                xi_D_hat[1] = fft(xi_D[1])
                
                #Solve convolutional inverse
                d_hat = solve_conv_term_D(zhat_mat, zhat_inv_mat, xi_D_hat, rho, size_z)
                d = np.real(ifft(d_hat))                      
                
                if (i_d == max_it_d - 1):
                    obj_val = obj_func(z_hat, d_hat, b,
                                       lambda_residual, lambda_prior, 
                                       psf_radius, size_z, size_x)
                    
                    print('--> Obj %3.3f'% obj_val)
            
                        
                obj_val_filter = obj_val
                
                #Debug progress
                d_diff = d - d_old
                d_comp = d
        
                if (i_d == max_it_d - 1):
                    obj_val = obj_func(z_hat, d_hat, b,
                                       lambda_residual, lambda_prior, 
                                       psf_radius, size_z, size_x)
                    print('Iter D %d, Obj %3.3f, Diff %5.5f'%
                          (i, obj_val, linalg.norm(d_diff) / linalg.norm(d_comp)))
               
               
        """Z-STEP"""
        #Precompute what is necessary for later
        [dhat_flat, dhatTdhat_flat] = precompute_Z_step(d_hat, size_x)        
        dhatT_flat = np.ma.conjugate(dhat_flat.T)
        
        z_old = z
               
        for i_z in range(max_it_z):
            
            #Compute v = [Dz,z]
            d_hat_dot_z_hat = np.multiply(d_hat, z_hat)
            v_Z[0] = np.real(ifft(np.sum(d_hat_dot_z_hat, axis=1).reshape(size_x)))
            v_Z[1] = z

            #Compute proximal updates
            u = v_Z[0] - d_Z[0]
            theta = lambdas[0]/gammas_Z[0]
            u_Z[0] = np.divide((Mtb + 1.0/theta * u ), (M + 1.0/theta * np.ones(size_x)))
            
            u = v_Z[1] - d_Z[1]
            theta = lambdas[1]/gammas_Z[1] * np.ones(u.shape)                        
            u_Z[1] = np.multiply(np.maximum(0, 1 - np.divide(theta, np.abs(u))), u)

            #Update Langrange multipliers
            d_Z[0] = d_Z[0] + (u_Z[0] - v_Z[0])
            d_Z[1] = d_Z[1] + (u_Z[1] - v_Z[1])
            
            #Compute new xi=u+d and transform to fft
            xi_Z[0] = u_Z[0] + d_Z[0]
            xi_Z[1] = u_Z[1] + d_Z[1]

            xi_Z_hat[0] = fft(xi_Z[0])
            xi_Z_hat[1] = fft(xi_Z[1])
            
            #Solve convolutional inverse
            z_hat = solve_conv_term_Z(dhatT_flat, dhatTdhat_flat, xi_Z_hat, gammas_Z, size_z)
            z = np.real(ifft(z_hat))
            
            if (i_z == max_it_z - 1):
                obj_val = obj_func(z_hat, d_hat, b,
                                   lambda_residual, lambda_prior, 
                                   psf_radius, size_z, size_x)
                
                print('--> Obj %3.3f'% obj_val)
                    
        obj_val_z = obj_val
        
        list_obj_val[i] = obj_val
        list_obj_val_filter[i] = obj_val_filter
        list_obj_val_z[i] = obj_val_z
               
        #Debug progress
        z_diff = z - z_old
        z_comp = z
        
        print('Iter Z %d, Obj %3.3f, Diff %5.5f'%
              (i, obj_val, linalg.norm(z_diff) / linalg.norm(z_comp)))
        
        #Termination
        if (linalg.norm(z_diff) / linalg.norm(z_comp) < tol and
            linalg.norm(d_diff) / linalg.norm(d_comp) < tol):
            break
        
    """Final estimate"""
    z_res = z
    
    d_res = d
    d_res = np.roll(d_res, psf_radius, axis=1)
    d_res = d_res[:, 0:psf_radius*2+1] 
    
    fft_dot = np.multiply(d_hat, z_hat)
    Dz = np.real(ifft(np.sum(fft_dot, axis=1).reshape(size_x)))
    
    obj_val = obj_func(z_hat, d_hat, b,
                       lambda_residual, lambda_prior, 
                       psf_radius, size_z, size_x)
    print('Final objective function %f'% (obj_val))
    
    reconstr_err = reconstruction_err(z_hat, d_hat, b, psf_radius, size_x)
    print('Final reconstruction error %f'% reconstr_err)
    
    return [d_res, z_res, Dz, list_obj_val, list_obj_val_filter, list_obj_val_z, reconstr_err]
        

def KernelConstraintProj(u, size_k_full, psf_radius):
    """Computes the proximal operator for kernel by projection"""

    #Get support
    u_proj = u
    u_proj = np.roll(u_proj, psf_radius, axis=1)
    u_proj = u_proj[:, 0:psf_radius*2+1]
    
    #Normalize
    u_sum = np.sum(np.power(u_proj, 2), axis=1)
    u_norm = np.tile(u_sum, [u_proj.shape[1], 1]).transpose(1, 0)
    u_proj[u_norm >= 1] = u_proj[u_norm >= 1] / np.sqrt(u_norm[u_norm >= 1])
    
    #Now shift back and pad again   
    u_proj = np.pad(u_proj, ((0,0),
                             (0, size_k_full[1] - (2*psf_radius+1))),
                    mode='constant', constant_values=0)
       
    u_proj = np.roll(u_proj, -psf_radius, axis=1)

    return u_proj

def precompute_D_step(z_hat, size_z, rho):
    """Computes to cache the values of Z^.T and (Z^.T*Z^ + rho*I)^-1 as in algorithm"""
    
    n = size_z[0]
    k = size_z[1]
    
    zhat_mat = np.transpose(z_hat, [2, 0, 1])
    zhat_inv_mat = np.zeros((zhat_mat.shape[0], k, k), dtype=imaginary_type)
    inv_rho_z_hat_z_hat_t = np.zeros((zhat_mat.shape[0], n, n), dtype=imaginary_type)
    z_hat_mat_t = np.transpose(np.ma.conjugate(zhat_mat), [0, 2, 1])
    
    #Compute Z_hat * Z_hat^T for each pixel
    z_hat_z_hat_t = np.einsum('knm,kmj->knj',zhat_mat, z_hat_mat_t)
    
    for i in range(zhat_mat.shape[0]):
        z_hat_z_hat_t_plus_rho = z_hat_z_hat_t[i]
        z_hat_z_hat_t_plus_rho.flat[::n + 1] += rho
        inv_rho_z_hat_z_hat_t[i] = linalg.pinv(z_hat_z_hat_t_plus_rho)
                                 
    zhat_inv_mat = 1.0/rho * (np.eye(k) - 
                              np.einsum('knm,kmj->knj',                              
                                        np.einsum('knm,kmj->knj',
                                                  z_hat_mat_t,
                                                  inv_rho_z_hat_z_hat_t),
                                        zhat_mat))
    
    print ('Done precomputing for D')
    return [zhat_mat, zhat_inv_mat]


def precompute_Z_step(dhat, size_x):
    """Computes to cache the values of D^.T and D^.T*D^ as in algorithm"""

    dhat_flat = dhat.T
    dhatTdhat_flat = np.sum(np.multiply(np.ma.conjugate(dhat_flat), dhat_flat), axis=1)
    print ('Done precomputing for Z')
    return [dhat_flat, dhatTdhat_flat]


def solve_conv_term_D(zhat_mat, zhat_inv_mat, xi_hat, rho, size_z):
    """Solves argmin(||Zd - x1||_2^2 + rho * ||d - x2||_2^2"""    

    k = size_z[1]
    sx = size_z[2]

    #Reshape to array per frequency  
    xi_hat_0_flat = np.expand_dims(xi_hat[0].T, axis=2)
    xi_hat_1_flat = np.expand_dims(xi_hat[1].T, axis=2)
    
    x = np.zeros((zhat_mat.shape[0], k), dtype=imaginary_type)
    z_hat_mat_t = np.ma.conjugate(zhat_mat.transpose(0,2,1))
    x = np.einsum("ijk, ikl -> ijl", zhat_inv_mat,
                                     np.einsum("ijk, ikl -> ijl",
                                               z_hat_mat_t,
                                               xi_hat_0_flat) +
                                     rho * xi_hat_1_flat) \
                  .reshape(sx, k)
        
    #Reshape to get back the new D^
    d_hat = x.T
    
    return d_hat
    
    
def solve_conv_term_Z(dhatT, dhatTdhat, xi_hat, gammas, size_z ):
    """Solves argmin(||Dz - x1||_2^2 + rho * ||z - x2||_2^2"""    

    sx = size_z[2]
    rho = gammas[1]/gammas[0]
    
    #Compute b       
    xi_hat_0_rep = np.expand_dims(xi_hat[0], axis = 1)
    xi_hat_1_rep = xi_hat[1]
    
    b = np.multiply(dhatT, xi_hat_0_rep) + rho * xi_hat_1_rep
    
    scInverse = np.divide(np.ones((1, sx)), rho*np.ones((1, sx)) + dhatTdhat.T)
    
    dhatT_dot_b = np.multiply(np.ma.conjugate(dhatT), b)
    dhatTb_rep = np.expand_dims(np.sum(dhatT_dot_b, axis=1), axis=1)
    x = 1.0/rho * (b - np.multiply(np.multiply(scInverse, dhatT), dhatTb_rep))
    
    z_hat = x
    return z_hat

      
def obj_func(z_hat, d_hat, b, lambda_residual, lambdas, \
                      psf_radius, size_z, size_x):
    """Computes the objective function including the data-fitting and regularization terms"""
    #Data-fitting term
    d_hat_dot_z_hat = np.multiply(d_hat, z_hat)
    Dz = np.real(ifft(np.sum(d_hat_dot_z_hat, axis=1).reshape(size_x)))
    
    f_z = lambda_residual * 1.0/2.0 * \
          np.power(linalg.norm(np.reshape(Dz[:, psf_radius:(Dz.shape[1] - psf_radius)] - b, \
                                          -1, 1)), 2)
    #Regularizer
    z = ifft(z_hat)
    g_z = lambdas * np.sum(np.abs(z))
    
    f_val = f_z + g_z
    return f_val
    
    
def reconstruction_err(z_hat, d_hat, b, psf_radius, size_x):
    """Computes the reconstruction error from the data-fitting term"""
    
    d_hat_dot_z_hat = np.multiply(d_hat, z_hat)
    Dz = np.real(ifft(np.sum(d_hat_dot_z_hat, axis=1).reshape(size_x)))
    
    err = 1.0/2.0 * \
          np.power(linalg.norm(np.reshape(Dz[:, psf_radius:(Dz.shape[1] - psf_radius)] - b, \
                                          -1, 1)), 2)
                                          
    return err