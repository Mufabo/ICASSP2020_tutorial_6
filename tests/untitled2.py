import numpy as np
import matplotlib.pyplot as plt

data = np.load('./tests/data_test.npy')

K_true = 3

r = 2



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
