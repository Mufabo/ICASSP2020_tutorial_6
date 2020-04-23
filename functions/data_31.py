import numpy as np

def data_31(N_k, epsilon):
    """
    Creates data based on http://arxiv.org/pdf/1811.12337v1

    Args:
        N_k : int, number of data vectors per cluster
        epsilon : float, percentage of replacement outliers

    Returns:
        data : 2darray of size N x (r+1) . data[, 0] includes labels from 1 to K_true               +1, where K_true+1 are the outliers, data[, 1:(r+1)] includes the actual data points
        r : int, number of features/dimensions in the generated data set
        N : int, total number of samples in the data set
        K_true : int, true numnber of clusters in the data set
        mu_true : 2darray of size r x K_true, true mean values
        scatter_true : 3darray of size r x rx K_true, true scatter matrices
    """

    out_range = np.array([[-20, 20], [-20, 20]])
    K_true = 3 # number of clusters
    r = 2 # number of features/dimensions

    mu_true = np.array([[0, 5, -5],
                        [5, 0, 0])

    scatter_true = np.array([[[2, 0.5], [0.5, 0.5]],
                             [[1, 0], [0, 0.1]],
                             [[2, -0.5], [-0.5, 0.5]]])

    N = K_true * N_K # total number of data points

    data = np.zeros([N, r+1])
    for k in range(K_true):
        
