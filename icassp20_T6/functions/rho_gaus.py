import numpy as np

def rho_gaus(t, r):
    """
    Computes rho(t) of the Gaussian distribution
    
    Args:
        t : 1darray of size N, Mahalanobis distances
        r : int, dimension
        
    Returns:
        rho : 1darray of size N, rho(t) of the Gaussian distribution
    """
    return r/2*np.log(2*np.pi) + t/2
    
    
