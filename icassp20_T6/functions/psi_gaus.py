import numpy as np
"""
Computes psi(t) of the Gaussian distribution.
Args:
    t : 1darray, squared Mahalanobis distances
"""
psi_gaus = lambda t: 0.5 * np.ones(len(t))

