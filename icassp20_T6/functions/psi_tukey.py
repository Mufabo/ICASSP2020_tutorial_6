import numpy as np

def psi_tukey(t, cT):
    """
    Computes psi(t) of the the Tukey loss function
    
    Args:
        t : 1darray of size N, Mahalanobis distances
        cT : float, tuning parameter
        
    Returns:
        psi : 1darray of size N, psi of Tukey loss function
    """
    
    psi = np.zeros(len(t))
    psi[t <= cT**2] = 0.5 * (1 - t[t<=cT**2]/cT**2)**2
    
    return psi
