"""
Computes eta of t distribution

Args:
    t : 1darray, squared Mahalanobs distances
    r : int, dimension
    nu : int, degree of freedom
"""
eta_t = lambda t, r, nu: -0.5 * (nu + r) / (nu + t)**2