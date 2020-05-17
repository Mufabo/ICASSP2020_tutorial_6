import numpy as np
import icassp20_T6 as t6
import warnings

def BIC_F(data, S_est, mu_est, t, mem, rho, psi, eta):
    """
    Computes the BIC of a RES distribution based on a finite sample
    penalty term
    
    Args:
        data : 2darray of shape (N, r)

        
        S_est : 3darray of shape (ll, r, r) Estimated scatter matrix of
            all clusters
            
        mu_ests : 2darray of shape (ll, r). Estimated mean values of all
            clusters
            
        t : 2darray of shape (N, ll). Squared Mahalanobis distances of data
            points in cluster m
        
        mem : 2darray of shape (N, ll) cluster memberships represented
            as matrix of one-hot rows
            
        rho : rho of density generator g
        psi : psi of density generator g
        eta : eta of density generator g
        
    Returns:
        bic : float, bic
        pen : float, penalty term
        like : float, likelihood term
    """
    
    N_m = np.sum(mem, axis=0)
    ll, _, r = S_est.shape
    D = t6.duplicationMatrix(r)
    
    q = 1/2*r*(r+3)
    
    temp_rho = np.zeros(ll)
    logdetS = np.zeros(ll)
    detJ = np.zeros(ll)
    
    for m in range(ll):
        x_hat_m = data[mem[:, m]] - mu_est[m]
        t_m = t[mem[:, m], m]
        J = t6.FIM_RES(x_hat_m, t_m , S_est[m,:,:], psi, eta, D)
        detJ[m] = np.linalg.det(J)
        temp_rho[m] = np.sum(rho(t[mem[:,m], m]))
        logdetS[m] = np.log(np.linalg.det(S_est[m,:,:]))
        
        if detJ[m] < 0:
            warnings.warn("negative determinant, J still not positive definite")
            detJ[m] += 10**-10
            if detJ[m] < 0:
                detJ[m] = np.abs(detJ[m])
        elif detJ[m] == 0 and N_m[m] == 0:
            warnings.warn("cluster without data point, zero determinant")
            detJ[m] = 1
        elif detJ[m] == 0:
            warnings.warn("zero determinant")
            detJ[m] += 10**-10
    
    like = -np.sum(temp_rho) + np.sum(N_m[N_m>0] * np.log(N_m[N_m > 0]), axis=0) \
        - np.sum(N_m[N_m > 0] * logdetS[N_m > 0], axis = 0)/2
            
    pen = -.5 * np.sum(np.log(detJ)) + ll * q/2 * np.log(2*np.pi) - ll * np.log(ll)
    
    bic = like + pen
    
    return bic, like, pen