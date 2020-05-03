import numpy as np
from scipy.special import gamma

def g_t(t, r, nu):
    """
    Computes g(t) of the t distribution
    
    Args:
        t : 1darray of size N, squared Mahalanobis distance
        r : int, dimension
        nu : int, degree of freedom
        
    Returns:
        res : 1darray of size N, g(t) of the t distribution
    """
    return gamma((nu + r)/2) / (gamma(nu/2)*(np.pi*nu)**(r/2)) * (1 + t/nu)**(-(nu+r)/2)
