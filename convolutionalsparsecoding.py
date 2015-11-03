import numpy as np
from numpy import linalg
from scipy.fftpack import fft2, ifft2

def learn_conv_sparse_coder(b, size_kernel, lambda_residual, lambda_prior, max_it, tol):
    # Parameters for this function
    #   b               : the image dataset with size (num_images, height, width)
    #   size_kernel     : the size of each kernel (num_kernels, height, width)
    #   lambda_residual :
    #   lambda_prior    :
    #   max_it          : the maximum iterations of the outer loop
    #   tol             : the minimum amount of changes in filters and codes to continue the algorithm
    
    real_type       = 'float64'
    imaginary_type  = 'complex128'
    
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
               
    
    #Pack lambdas and find algorithm params
    lambdas = [lambda_residual, lambda_prior]
    
    gamma_heuristic = 60 * lambda_prior * 1/np.amax(b)
    gammas_D = [gamma_heuristic / 5000, gamma_heuristic] #[gamma_heuristic / 2000, gamma_heuristic];    
    gammas_Z = [gamma_heuristic / 500, gamma_heuristic] #[gamma_heuristic / 2, gamma_heuristic];
    
    #Initialize variables for K
    varsize_D = [size_x, size_k_full]
    xi_D     = [np.zeros(varsize_D[0], dtype = real_type),
                np.zeros(varsize_D[1], dtype = real_type)]
                
    xi_D_hat = [np.zeros(varsize_D[0], dtype = imaginary_type),
                np.zeros(varsize_D[1], dtype = imaginary_type)]
    
    u_D = [np.zeros(varsize_D[0], dtype = real_type),
           np.zeros(varsize_D[1], dtype = real_type)]
           
    d_D = [np.zeros(varsize_D[0], dtype = real_type),
           np.zeros(varsize_D[1], dtype = real_type)]
           
    v_D = [np.zeros(varsize_D[0], dtype = real_type),
           np.zeros(varsize_D[1], dtype = real_type)]
    
    #Initial iterates
    d = np.random.normal(size=size_kernel)
    d = np.pad(d, ((0, 0),
                   (0, size_x[1] - size_kernel[1]),
                   (0, size_x[2] - size_kernel[2])),
               mode='constant', constant_values=0)
    d = np.roll(d, -int(psf_radius), axis=1)
    d = np.roll(d, -int(psf_radius), axis=2)
    d_hat = fft2(d)
    
    #Initialize variables for Z
    varsize_Z = [size_x, size_z]
    xi_Z = [np.zeros(varsize_Z[0], dtype = real_type),
            np.zeros(varsize_Z[1], dtype = real_type)]
            
    xi_Z_hat = [np.zeros(varsize_Z[0], dtype = imaginary_type),
                np.zeros(varsize_Z[1], dtype = imaginary_type)]
    
    u_Z = [np.zeros(varsize_Z[0], dtype = real_type),
           np.zeros(varsize_Z[1], dtype = real_type)]
           
    d_Z = [np.zeros(varsize_Z[0], dtype = real_type),
           np.zeros(varsize_Z[1], dtype = real_type)]
           
    v_Z = [np.zeros(varsize_Z[0], dtype = real_type),
           np.zeros(varsize_Z[1], dtype = real_type)]
    
    #Initial iterates(change if use with known z)    
    z = np.random.normal(size=size_z)
    z_hat = fft2(z)
    
    #Initial vals
    obj_val = obj_func(z_hat, d_hat, b,
                       lambda_residual, lambda_prior,
                       psf_radius, size_z, size_x)
        
    #Iteration for local back and forth
    max_it_d = 10
    max_it_z = 10
    
    obj_val_filter = obj_val
    obj_val_z = obj_val
    
    #Start the main algorithm
    for i in range(max_it):
        #Update kernels
        #Recompute what is necessary for kernel convterm later
        rho = gammas_D[1]/gammas_D[0]
        [zhat_mat, zhat_inv_mat] = precompute_H_hat_D(z_hat, rho, size_z)
        
        obj_val_min = min(obj_val_filter, obj_val_z)
        
        d_old = d
        d_hat_old = d_hat
        
        for i_d in range(max_it_d):
           
            #Compute v_i = H_i * z                      
            d_hat_dot_z_hat = np.multiply(d_hat, z_hat)
            v_D[0] = np.real(ifft2(np.sum(d_hat_dot_z_hat, axis=1).reshape(size_x)))
            v_D[1] = d

            
            #Compute proximal updates
            u = v_D[0] - d_D[0]        
            theta = lambdas[0]/gammas_D[0]
            u_D[0] = np.divide((Mtb + 1.0/theta * u), (M + 1.0/theta * np.ones(size_x)))
            
            u = v_D[1] - d_D[1]
            u_D[1] = kernel_constraint_proj(u, size_k_full, psf_radius)
            
            #Update running errors
            d_D[0] = d_D[0] - (v_D[0] - u_D[0])
            d_D[1] = d_D[1] - (v_D[1] - u_D[1])
            
            #Compute new xi and transform to fft
            xi_D[0] = u_D[0] + d_D[0]
            xi_D[1] = u_D[1] + d_D[1]
            xi_D_hat[0] = fft2(xi_D[0])
            xi_D_hat[1] = fft2(xi_D[1])
            
            
            #Solve convolutional inverse
            #d = ( sum_j(gamma_j * H_j'* H_j) )^(-1) * ( sum_j(gamma_j * H_j'* xi_j) )
            d_hat = solve_conv_term_D(zhat_mat, zhat_inv_mat, xi_D_hat, rho, size_z)
            d = np.real(ifft2(d_hat))
                       
            #d_matlab = np.zeros(d.shape)
            #fh.read_file('/home/chau/Dropbox/PRIM/d.txt', d_matlab, 3)
            
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
        print('Iter D %d, Obj %3.3g, Diff %5.5g'%
              (i, obj_val, linalg.norm(d_diff) / linalg.norm(d_comp)))
               
        #Update sparsity term
        
        #Recompute what is necessary for convterm later
        [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(d_hat, size_x)        
        dhat_flat_conj = np.ma.conjugate(dhat_flat.T)
        
        z_old = z
        z_hat_old = z_hat
        
        for i_z in range(max_it_z):
            
            #Compute v_i = D_i * z            
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
            d_Z[0] = d_Z[0] - (v_Z[0] - u_Z[0])
            d_Z[1] = d_Z[1] - (v_Z[1] - u_Z[1])
            
            #Compute new xi and transform to fft
            xi_Z[0] = u_Z[0] + d_Z[0]
            xi_Z[1] = u_Z[1] + d_Z[1]

            xi_Z_hat[0] = fft2(xi_Z[0])
            xi_Z_hat[1] = fft2(xi_Z[1])
            
            #Solve convolutional inverse
            # z = ( sum_j(gamma_j * H_j'* H_j) )^(-1) * ( sum_j(gamma_j * H_j'* xi_j) )
            z_hat = solve_conv_term_Z(dhat_flat_conj, dhatTdhat_flat, xi_Z_hat, gammas_Z, size_z)
            z = np.real(ifft2(z_hat))
                       
            obj_val = obj_func(z_hat, d_hat, b,
                               lambda_residual, lambda_prior, 
                               psf_radius, size_z, size_x)
            
            print('--> Obj %3.3f'% obj_val)
        
        obj_val_z = obj_val
        
        #Stoping criteria
        if (obj_val_min <= obj_val_filter and obj_val_min <= obj_val_z):
            z_hat = z_hat_old
            z = np.real(ifft2(z_hat))
            
            d_hat = d_hat_old
            d = np.real(ifft2(d_hat))
            
            obj_val = obj_func(z_hat, d_hat, b,
                               lambda_residual, lambda_prior, 
                               psf_radius, size_z, size_x)
            break
        
        z_diff = z - z_old
        z_comp = z
        print('Iter Z %d, Obj %3.3f, Diff %5.5g'%
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
    
    return [d_res, z_res, Dz]
        

def kernel_constraint_proj(u, size_k_full, psf_radius):
    
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

def precompute_H_hat_D(z_hat, rho, size_z):
    
    #Computes the spectra for the inversion of all H_i
    
    #Size
    n = size_z[0]
    k = size_z[1]
    sy = size_z[2]
    sx = size_z[3] 
    
    #Precompute spectra for H    
    zhat_mat = np.ndarray.transpose(z_hat.transpose(0,1,3,2).reshape(n, k, -1), [2, 0, 1])

    #Precompute the inverse matrices for each frequency
    zhat_inv_mat = np.zeros((zhat_mat.shape[0], k, k), dtype='complex128')
    
    
    #Not sure if this part could be accelerated or not
    for i in range(sy * sx):
        
        z_hat_mat_t = np.ma.conjugate(zhat_mat[i,:]).T
        
        zhat_inv_mat[i,:] = 1.0/rho * \
                (np.eye(k) -
                np.dot(np.dot(z_hat_mat_t,
                              np.linalg.pinv(rho * np.eye(n) +
                                             np.dot(zhat_mat[i,:], z_hat_mat_t))),
                       zhat_mat[i,:]))
                               
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

    #Reshape to array per frequency  
    xi_hat_0_flat = np.expand_dims(np.reshape(xi_hat[0].transpose(0,2,1),
                                              (n, sx * sy)).T,
                                   axis=2)
    xi_hat_1_flat = np.expand_dims(np.reshape(xi_hat[1].transpose(0,2,1),
                                              (k, sx * sy)).T,
                                   axis=2)
    
    #Invert
    x = np.zeros((zhat_mat.shape[0], k), dtype='complex128')
    z_hat_mat_t = np.ma.conjugate(zhat_mat.transpose(0,2,1))
    x = np.einsum("ijk, ikl -> ijl", zhat_inv_mat,
                  np.einsum("ijk, ikl -> ijl", z_hat_mat_t, xi_hat_0_flat) +
                  rho * xi_hat_1_flat).reshape(sx * sy, k)
        
    #Reshape to get back the new Dhat
    d_hat = np.reshape(x.T, (k, sy, sx)).transpose(0,2,1)
    
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
    
    #sum all the kernels
    x = 1.0/rho * (b - np.multiply(np.multiply(scInverse, dhatT), dhatT_dot_b))
    
    #Final transpose gives z_hat
    z_hat = x.reshape(size_z).transpose(0,1,3,2)
    
    return z_hat
    
def obj_func(z_hat, d_hat, b, lambda_residual, lambdas, \
                      psf_radius, size_z, size_x):
    
    #Dataterm and regularizer    
    #d_hat_rep = np.tile(d_hat, [size_z[0],1,1,1])
    d_hat_dot_z_hat = np.multiply(d_hat, z_hat)
    Dz = np.real(ifft2(np.sum(d_hat_dot_z_hat, axis=1).reshape(size_x)))
    
    f_z = lambda_residual * 1.0/2.0 * \
          np.power(linalg.norm(np.reshape(Dz[:, psf_radius:(Dz.shape[1] - psf_radius), \
                                                psf_radius:(Dz.shape[2] - psf_radius)] - b, \
                                          -1, 1)), 2)
    
    z = ifft2(z_hat)
    g_z = lambdas * np.sum(np.abs(z))
    
    #Function val
    f_val = f_z + g_z
    
    return f_val