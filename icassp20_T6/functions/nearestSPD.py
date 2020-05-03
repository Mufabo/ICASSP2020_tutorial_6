import numpy as np

def nearestSPD(A):
    """
    Finds the nearest symmetric, positive definite matrix to A in the 
    Frobenius norm
    
    Args:
        A : 2darray. square matrix which will be converted to the
            nearest spd matrix
            
    Returns:
        Ahat : 2darray. Nearest SPD matrix to A
    
    Raises:
        ValueError if A is not square
        
    References:
        http://www.sciencedirect.com/science/article/pii/0024379588902236
    """
    
    r, c = A.shape
    
    if r != c:
        raise ValueError("Matrix A is not square")

    
    # symmetrize A into B
    B = (A + A.T)/2
    
    # Compute the symmetric polar factor of B, called H
    # H is spd
    U, Sigma, V = np.linalg.svd(B)
    H = V.T @ np.diag(Sigma) @ V
    
    # Get Ahat in the above formula
    Ahat = (B + H)/2
    
    # ensure symmetry
    Ahat = (Ahat + Ahat.T)/2
    
    # check if Ahat is spd
    eig_vals = np.linalg.eigvals(Ahat)
    k = 1
    while not all(eig_vals) >= 0:
        mineig = np.min(eig_vals)
        Ahat += (-mineig * k**2 + np.spacing(mineig) + 10**-10) * np.eye(len(A))
        eig_vals = np.linalg.eigvals(Ahat)
        k += 1
        
    return Ahat
        
    