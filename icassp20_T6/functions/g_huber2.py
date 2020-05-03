import numpy as np
from scipy.stats.distributions import chi2
from scipy.special import gamma
from scipy.special import gammaincc

def g_huber2(t, r, qH=0.8, cH=None, bH=None, aH=None):
    """
    Computes g(t) of the Huber distribution
    
    Possible input combinations:
        t, r
        t, r, qH
        t, r, cH, bH, aH . This option is provided to improve performance
        because it allows to avoid the calculation of the constants cH, bH and
        aH in every loop iteration
    
    Args:
        t : 1darray of size N, squared Mahalanobis distances
        r : int, dimension
        qH : float, tuning parameter, standard value 0.8, choose qH > 0.701
        cH : float, tuning parameter
        bH : float, tuning parameter
        aH : float, tuning parameter
        
    Returns:
        g: 1darray of size N, g(t) of Huber distribution
        
    Raises:
        ValueError: If incorrect combination of inputs
    """

    if sum([s is None for s in [cH, bH, aH]]) != 3 and sum([s is None for s in [cH, bH, aH]]) != 0:
        raise ValueError("Incorrect combination of inputs")
        
    igamma = lambda a, b: gammaincc(a, b)* gamma(a)
    
    if sum([s is None for s in [cH, bH, aH]]) == 3:
        cH = np.sqrt(chi2.ppf(qH, r))
        bH = chi2.cdf(cH**2, r+2) + cH**2/r*(1-chi2.cdf(cH**2, r))
        aH = gamma(r/2) / np.pi**(r/2) / ( (2*bH)**(r/2) * (gamma(r/2) - igamma(r/2, cH**2 / (2*bH))) + (2*bH*cH**r*np.exp(-cH**2/(2*bH))) / (cH**2 - bH*r))

    g = np.zeros(len(t))
    g[t <= cH**2] = aH * np.exp(-t[t<=cH**2]/(2*bH))
    g[t > cH**2] = aH * (np.exp(1) * t[t > cH**2]/cH**2)**(-cH**2/(2*bH))
    
    return g

