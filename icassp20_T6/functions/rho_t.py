import numpy as np
from scipy.special import gamma

def rho_t(t, r, nu):
    """
    Computes rho(t) of the t distribution
    
    Args:
        t : 1darray of size N, squared Mahalanobis distance
        r : int, dimension
        nu : int, degree of freedom
        
    Returns:
        rho : 1darray of size N, rho(t) of the t distribution
    """
    return -np.log(gamma((nu + r)/2)) + np.log(gamma(nu/2)) + 0.5 * r * np.log(np.pi * nu) + (nu + r) / 2 * np.log(1 + t/nu)
