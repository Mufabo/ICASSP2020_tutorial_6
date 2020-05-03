import numpy as np

def data_31(N_k, epsilon):
    """
    Creates data based on http://arxiv.org/pdf/1811.12337v1

    Args:
        N_k : int, number of data vectors per cluster
        
        epsilon : float, percentage of replacement outliers

    Returns:
        data : 2darray of size N x (r+1) . data[, 0] includes labels from 1 to K_true+1, 
        where K_true+1 are the outliers, data[, 1:(r+1)] includes the actual data points
        
        r : int, number of features/dimensions in the generated data set
        
        N : int, total number of samples in the data set
        
        K_true : int, true numnber of clusters in the data set
        
        mu_true : 2darray of size r x K_true, true mean values
        
        scatter_true : 3darray of size r x r x K_true, true scatter matrices
    """

    out_range = np.array([[-20, 20], [-20, 20]])
    K_true = 3 # number of clusters
    r = 2 # number of features/dimensions

    mu_true = np.array([[0, 5, -5], [5, 0, 0]])

    scatter_true = np.array([[[2, 0.5], [0.5, 0.5]], [[1, 0], [0, 0.1]], [[2, -0.5], [-0.5, 0.5]]])

    N = K_true * N_k # total number of data points

    data = np.zeros([N, r+1])
    for k in range(K_true):
      data[(k*N_k):((k+1)*N_k)] = np.hstack([np.ones([N_k, 1])*(k+1), np.random.multivariate_normal(mu_true[:, k], scatter_true[k, :, :], N_k)])
    
    # randomly permutate data
    data = data[np.random.permutation(N), :]
    
    # replacement outlier
    N_repl = int(np.round(N * epsilon))
    index_repl = np.random.permutation(N)[:N_repl]
    
    data_rpl = np.zeros([N_repl, r])
    for ir in range(r):
      data_rpl[:, ir] = np.random.rand(N_repl) * (out_range[ir, 1] - out_range[ir, 0] + out_range[ir, 1])

    data[index_repl, :] = np.vstack([np.ones([N_repl, 1])*(K_true+1), data_rpl])
      
    
    return data, r, N, K_true, mu_true, scatter_true
    
    
  
        
