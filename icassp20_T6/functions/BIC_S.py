import numpy as np

def BIC_S(S_est, t, mem, rho):
    """
    Computes the BIC of a RES distribution with Schwarz penalty term
    
    Args:
        S_est : 3darray of shape (ll, r, r), estimated scatter matrix
            of cluster m
        t : 2darray of shape (N, ll), squared mahalanobis distances
            of data points in cluster m
        mem : 2darray of shape (N, ll), cluster memberships
        rho : rho of density generator g
        
    Returns:
        bic : bic
        pen : penalties
        like: likelihood
    """
    
    N_m = np.sum(mem, axis=0)
    ll, _, r = S_est.shape
    q = .5 * r * (r+3)
    N = 1 if len(t) == 0 else len(t)
    
    temp_rho = np.zeros(ll)
    logdetS = np.zeros(ll)
    
    for m in range(ll):
        temp_rho[m] = np.sum(rho(t[mem[:,m], m]))
        logdetS[m] = np.log(np.linalg.det(S_est[m,:,:]))
        
    like = -np.sum(temp_rho) + np.sum(N_m[N_m>0] * np.log(N_m[N_m>0]), axis=0) \
        - np.sum(N_m[N_m>0] * logdetS[N_m > 0], axis = 0)/2
    
    pen = -q*ll/2*np.log(N)
        
    bic = like + pen

    return bic, like, pen 
    