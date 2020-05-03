import numpy as np
import icassp20_T6 as t6

def FIM_RES(x_hat_m, t_m, S_est, psi, eta, D):
    """
    Computes FIM of one cluster for a given RES distribution
    
    Args:
        x_hat_m : 2darray of shape (N_m, r), data matrix. Note that in the
            Matlab version the dimensions are flipped
        t_m : 1darray of size N_m, squared Mahalanobis distances of
            data points in cluster m
        S_est : 2darray of shape (r, r) Estimated scatter matrix of
            cluster m
        psi : psi of density generator g
        eta : eta of density generator g
        D : 2darray of shape (r^2, .5*r*(r+1)) duplication matrix
        
    Returns:
        J : 2darray of shape (.5*r*(r+3), .5*r*(r+3)) FIM
    """
    
    r = len(S_est)
    N_m = len(t_m)
    
    # F_mumu
    temp_eta = np.zeros([N_m, r, r])  

    for n in range(N_m):
        temp_eta[n, :, :] = eta(t_m[n:n+1]) * x_hat_m[n:n+1].T @ x_hat_m[n:n+1]
    
    
    F_mumu = -4 * t6.mldivide(t6.mldivide(np.sum(temp_eta, axis=0), S_est), S_est) \
        - t6.mldivide(np.eye(r), S_est) * np.sum(psi(t_m),axis=0) * 2
        
    # F_muS
    temp_eta = np.zeros([N_m, r, r**2])
    for n in range(N_m):
        tmp_0 = t6.mldivide(x_hat_m[n:n+1].T @ x_hat_m[n:n+1], S_est)
        tmp_1 = t6.mldivide(tmp_0, S_est)
        tmp_2 = np.linalg.solve(S_est, x_hat_m[n])
        temp_eta[n,:,:] = eta(t_m[n:n+1]) * np.kron(tmp_1, tmp_2)
    
    
    F_muS = -2 * np.sum(temp_eta, axis=0) @ D
    
    # F_Smu
    F_Smu = F_muS.T
    
    # F_SS
    temp_eta = np.zeros([N_m, r**2, r**2])
    for n in range(N_m):
        tmp_0 = t6.mldivide(x_hat_m[n:n+1].T @ x_hat_m[n:n+1], S_est)
        tmp_1 = t6.mldivide(tmp_0, S_est)
        temp_eta[n,:,:] = eta(t_m[n:n+1]) * np.kron(tmp_1, tmp_1)
    
    F_SS = -D.T @ np.sum(temp_eta, axis = 0) @ D \
        - N_m/2 * D.T @ (t6.mldivide(np.eye(r**2), np.kron(S_est, S_est))) @ D 
    
    mat_1 = np.hstack([-F_mumu, -F_muS])
    mat_2 = np.hstack([-F_Smu, -F_SS])      
    J = np.vstack([mat_1, mat_2])
    
    """
    There are case where the FIM is not positive semidefinite.
    To catch these case, we use the function nearestSPD(), which
    calculates the nearest positive semidefinite matrix
    """
    if np.linalg.det(J) < 0:
        J = t6.nearestSPD(J)
        
    return J
    