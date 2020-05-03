import numpy as np

def duplicationMatrix(num_features):
    dup_mat_transpose = np.zeros([int(.5*num_features*(num_features+1)), num_features**2])
    
    for j in range(num_features):
        for i in range(num_features):

            u = np.zeros([int(.5*num_features*(num_features+1)), 1])
            idx = int(j * num_features + (i+1) - .5*j*(j+1)) - 1
            u[idx, 0] = 1
            
            Y = np.zeros([num_features, num_features])
            Y[i, j] = 1
            Y[j, i] = 1
            d = u @ np.reshape(Y, [1, -1], order='F')
            
            if j <= i:
                dup_mat_transpose += d
                
    return dup_mat_transpose.T