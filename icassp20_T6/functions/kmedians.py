import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


def kmedians(data, ll, max_iter = 10, replicates = 5):
    
    best = None
    for i in range(replicates):
        # Initialization using K-means++
        # returns list of arrays
        initial_centers = kmeans_plusplus_initializer(data, ll).initialize()
        for _ in range(max_iter):
            # Compute distances
            distances = np.zeros([len(data), ll])
            for l in range(ll):
                distances[:, l] = np.sum(np.abs(data - initial_centers[l]), axis=1)
                
            
        
        