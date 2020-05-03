import numpy as np

def eta_tukey(t, cT):
    """
    Computes eta(t) of the the Tukey loss function
    
    Args:
        t : 1darray of size N, Mahalanobis distances
        cT : float, tuning parameter
        
    Returns:
        psi : 1darray of size N, psi of Tukey loss function
    """
    
    eta = np.zeros(len(t))
    eta[t <= cT**2] = t[t <= cT**2]/cT**4 - 1/cT**2
    
    return eta

