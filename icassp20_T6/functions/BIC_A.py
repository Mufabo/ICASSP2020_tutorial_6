import numpy as np

def BIC_A(S_est, t, mem, rho, psi, eta):
    """
    Computes the BIC of a RES distribution with the asymptotic
    penalty term
    
    Args:
        S_est : 3darray of shape (ll, r, r) estimated scatter matrices of
            Cluster m
        t : 2darray of shape (N, ll) squared Mahalanobis distances of
            data points in cluster m
        mem : 2darray of shape (N, ll) cluster memberships
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
    q = .5*r*(r+3)
    
    temp_rho = np.zeros(ll)
    temp_psi = np.zeros(ll)
    temp_eta = np.zeros(ll)
    logdetS = np.zeros(ll)
    epsilon = np.zeros(ll)
    
    for m in range(ll):
        temp_rho[m] = np.sum(rho(t[mem[:,m], m]))
        temp_psi[m] = np.sum(psi(t[mem[:,m], m]))
        temp_eta[m] = np.sum(eta(t[mem[:,m], m]))
        
        epsilon[m] = np.max(np.array([np.abs(temp_psi[m]),
                                      np.abs(temp_eta[m]),
                                      N_m[m]]))
        
        logdetS[m] = np.log(np.linalg.det(S_est[m,:,:]))
        
    like = -np.sum(temp_rho[temp_rho>0]) + np.sum(N_m[N_m>0] \
        * np.log(N_m[N_m>0])) - np.sum(N_m * logdetS)/2
        
    pen = -.5 * q * np.sum(np.log(epsilon[epsilon > 0]), axis = 0)
    
    bic = like + pen
    
    return bic, like, pen