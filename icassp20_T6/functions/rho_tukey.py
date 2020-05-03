import numpy as np

def rho_tukey(t, r, cT):
    """
    Computes rho(t) of the the Tukey loss function
    
    Args:
        t : 1darray of size N, Mahalanobis distances
        r : int, dimension
        cT : float, tuning parameter
        
    Returns:
        rho : 1darray of size N, rho of Tukey loss function
    """
    
    rho = np.zeros(len(t))
    rho[t <= cT**2] = r/2*np.log(2*np.pi) + t[t <= cT**2]**3 /(6*cT**4) - t[t <= cT**2]**2 / (2*cT**2) + t[t<=cT**2]/2
    rho[t > cT**2] = r / 2 * np.log(2 * np.pi) + cT**2/6
    
    return rho
