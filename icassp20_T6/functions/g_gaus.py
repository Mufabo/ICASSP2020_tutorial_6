import numpy as np

def g_gaus(t, r, clip = 400):
    """
    Computes g(t) of the Gaussian distribution
    
    Args:
        t : 1darray of size N, squared Mahalanobis distance
        r : int, dimension
        clip : int, Clipping to avoid zero. Default = 400
        
    Returns:
        g : 1darray of size N, g(t) of Gaussian distribution
    """
       
    g = (2*np.pi)**(-r/2) * np.exp(-t/2)
    g[t >= clip] = (2*np.pi)**(-r/2) * np.exp(-clip/2)
    
    return g