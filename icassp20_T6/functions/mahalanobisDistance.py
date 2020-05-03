import numpy as np

def mahalanobisDistance(x, mu, S):
    """
    Computes the squared Mahalanobis distance
    
    Args:
        x : 2darray of shape (N, r). Data
        mu : 1darray of size r, cluster centroid
        S : 2darray of shape (r, r) cluster scatter matrix
        
    Returns:
        t : 1darray of size N. Squared Mahalanobis distance
    """
    mldivide = lambda A, B: np.linalg.lstsq(B.conj().T, A.conj().T, rcond=None)[0].conj().T
    return np.diag(mldivide((x.T - mu[:, None]).T, S) @ (x.T - mu[:, None]))