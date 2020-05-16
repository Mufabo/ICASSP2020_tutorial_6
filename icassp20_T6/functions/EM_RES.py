import numpy as np
import scipy.linalg as spl
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import warnings
import icassp20_T6 as t6

def EM_RES(data, ll, g, psi,limit = 1e-6, em_max_iter = 200, reg_value = 1e-6):
    """
    EM algorithm for mixture of RES distributions defined by g and psi
    
    Args:
        data : 2darray os shape (N, r). Data matrix without labels
        
        ll : int, number of clusters
        
        g : anonymous function of density generator
        
        psi : anonymous function of psi
        
        limit : float. Value that determines when the EM algorithm 
            should terminate
            
        em_max_iter : int. maximum number of iterations of the EM 
            algorithm
            
        reg_value : float. Regularization value used to regularize 
            the covariance matrix in the EM algorithm
        
    Returns:
        mu_hat : 2darray of shape(ll, r). Final estimate of cluster
            centroids.
            
        S_hat : 3darray of shape (ll, r, r). Final estimate of cluster
            scatter matrices.
            
        t : 2darray of shape (N, ll). Mahalanobis distances
        
        R : 2darray of shape (N, ll). Estimates of the posterior
            probabilities per cluster.
    """

    """
    Variable initialization
    """    
    N, r = np.shape(data)
    v = np.zeros([N, ll])
    v_diff = np.zeros([N, ll])
    tau = np.zeros(ll)
    S_hat = np.zeros([ll, r, r])
    t = np.zeros([N, ll])
    log_likelihood = np.zeros(em_max_iter)
    
    """
    Kmeans clustering
    """
    
    replicates = 500
    manhattan_metric = distance_metric(type_metric.MANHATTAN)
    best = None
    for i in range(replicates):
        # Initialization using K-means++
        initial_centers = kmeans_plusplus_initializer(data, ll).initialize()
        kmeans_instance = kmeans(data, initial_centers, itermax = 10                                , metric = manhattan_metric)
        kmeans_instance.process()
        error = kmeans_instance.get_total_wce()
        if best is None or error < best:
            best = error
            clu_memb_kmeans = kmeans_instance.get_clusters()
            mu_hat = np.array(kmeans_instance.get_centers())
            
    for m in range(ll):
        x_hat = data[clu_memb_kmeans[m], 0] - mu_hat[m][:, None]
        N_m = len(clu_memb_kmeans[m])
        tau[m] = N_m / N

        S_hat[m, :, :] = (x_hat @ x_hat.T) / N_m
        
        # Check if the sample covariance matrix is positive definite
        spd = all(spl.eigvals(S_hat[m, :, :]) > 0)
        # if not modify to get spd matrix
        if not spd or np.linalg.cond(S_hat[m, :, :]) > 30:
            S_hat[m, :, :] = 1/(r*N_m)*np.sum(np.diag(x_hat @ x_hat.T)) * np.eye(r) 
            if not all(spl.eigvals(S_hat[m, :, :]) > 0):
                S_hat[m, :, :] = np.eye(r)
        t[:, m] = t6.mahalanobisDistance(data, mu_hat[m], S_hat[m, :, :])
    
    """
    EM Algorithm
    """
    for ii in range(em_max_iter):
        # E-step
        v_lower = np.zeros([N, ll])
        for j in range(ll):                      
            v_lower[:, j] = tau[j] * np.linalg.det(S_hat[j,:,:])**-.5 * g(t[:, j])
            
        for m in range(ll):
            v[:, m] = tau[m] * np.linalg.det(S_hat[m,:,:])**-.5 * g(t[:, m]) / np.sum(v_lower, axis=1)
            v_diff[:, m] = v[:, m] * psi(t[:, m])

        # M-step
        for m in range(ll):
            mu_hat[m] = np.sum(v_diff[:,m:m+1] * data, axis=0) / np.sum(v_diff[:,m], axis=0)
            S_hat[m,:,:] = 2 * (v_diff[:,m] * (data - mu_hat[m]).T @ (data - mu_hat[m])) / np.sum(v[:,m], axis=0) + reg_value * np.eye(r)
            tau[m] = np.sum(v[:,m], axis=0)/N
            t[:,m] = t6.mahalanobisDistance(data, mu_hat[m], S_hat[m,:,:])

            
        # Check convergence
        v_conv = np.zeros([N, ll])
        for m in range(ll):
            v_conv[:, m] = tau[m] * np.linalg.det(S_hat[m,:,:])**-.5 * g(t[:,m])
        log_likelihood[ii] = np.sum(np.log(np.sum(v_conv, axis=1)) ,axis=0)
        
        if ii>1 and np.abs(log_likelihood[ii] - log_likelihood[ii-1]) < limit:
            break

    # Calculatate posterior probabilities
    R = v_conv / np.sum(v_conv, axis=1)[:, None]
    
    """
    Diagonal loading
    
    If the estimated "Matrix is close to singular or badly scaled" it cannot be inverted. 
    https://math.stackexchange.com/questions/261295/to-invert-a-matrix-condition-number-should-be-less-than-what
    If S_hat has a large condition number a small number in comparison to the
    matrix entries is added. This step should be subject for further tweaking.
    """
    for m in range(ll):
        cond_S = np.linalg.cond(S_hat[m,:,:])
        if 30 < cond_S:
            warnings.warn("S with large condition number")
            S_hat[m, :, :] += .01 * 10**np.floor(np.log10(np.trace(S_hat[m,:,:]))) * np.log10(cond_S) * np.eye(r)
    
    return mu_hat, S_hat, t, R