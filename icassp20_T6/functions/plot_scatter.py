import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(data, K_true, r):
    """
    Plots data with given labels for r in [1,2,3]
    
    Args:
        data : 2darray of size (N, r+1)
        K_true : int, true number of clusters
        r : int, dimension
    """
    
    if r==1 or r==2:
        for k in range(1, K_true+2):
            if r == 1:
                if k == K_true+1:
                    plt.scatter(data[data[:,0]==k, 1], np.zeros(np.sum(data[:,0]==k)), facecolor='none',edgecolor='k')
                else:
                   plt.scatter(data[data[:,0]==k, 1], np.zeros(np.sum(data[:,0]==k)))
            if r == 2:
                if k == K_true+1:
                    plt.scatter(data[data[:,0]==k, 1], data[data[:,0]==k, 2], s=15, facecolor='none',edgecolor='k')
                else:
                    plt.scatter(data[data[:,0]==k, 1], data[data[:,0]==k, 2], s=15)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')        
                    
    elif r == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if k == K_true+1:
            ax.scatter(data[data[:,0]==k, 1], data[data[:,0]==k, 2], data[data[:,0]==k, 3], s=15, facecolor='none',edgecolor='k')
        else:
            ax.scatter(data[data[:,0]==k, 1], data[data[:,0]==k, 2], data[data[:,0]==k, 3], s=15)
            
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.show()