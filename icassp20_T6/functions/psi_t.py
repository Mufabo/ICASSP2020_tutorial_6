def psi_t(t, r, nu):
    """
    Computes psi(t) of the t distribution
    
    Args:
        t : 1darray of size N, squared Mahalanobis distance
        r : int, dimension
        nu : int, degree of freedom
        
    Returns:
        psi : 1darray of size N, psi(t) of the t distribution
    """
    
    return 0.5 * (nu * r) / (nu + t)

