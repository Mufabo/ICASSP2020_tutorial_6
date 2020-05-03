import numpy as np
from scipy.stats.distributions import chi2


def eta_huber2(t, r, qH=0.8, cH=None, bH=None):
    """
    Computes eta(t) of the Huber distribution
    
    Possible input combinations:
        t, r
        t, r, qH
        t, r, cH, bH . This option is provided to improve performance
        because it allows to avoid the calculation of the constants cH, bH and
        aH in every loop iteration
    
    Args:
        t : 1darray of size N, squared Mahalanobis distances
        r : int, dimension
        qH : float, tuning parameter, standard value 0.8, choose qH > 0.701
        cH : float, tuning parameter
        bH : float, tuning parameter

        
    Returns:
        eta: 1darray of size N, eta(t) of Huber distribution
        
    Raises:
        ValueError: If incorrect combination of inputs
    """
    
    if sum([s is None for s in [cH, bH]]) != 2 and sum([s is None for s in [cH, bH]]) != 0:
        raise ValueError("Incorrect combination of inputs")
        
    if sum([s is None for s in [cH, bH]]) == 2:
        cH = np.sqrt(chi2.ppf(qH, r))
        bH = chi2.cdf(cH**2, r+2) + cH**2/r*(1-chi2.cdf(cH**2, r))
       
    eta = np.zeros(len(t))

    eta[t > cH**2] = -cH**2 / (2 * bH * t[t > cH**2]**2)
    return eta


