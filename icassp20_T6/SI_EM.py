import icassp20_T6 as t6
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

"""
Example for the EM algorithm. Compares the results of a Gaussian, t and
Huber based EM over different percentages of outliers.
"""

#%% User input

# percentage of replacement outliers
epsilon = .2

# Number of samples per cluster
N_k = 50

# Select degree of freedom for t distribution
nu = 3

# Select tuning parameter for Huber distribution
qH = .8

#create data
data, r, N, K_true, mu_true, S_true = t6.data_31(N_k, epsilon)

# Models
g = [partial(t6.g_gaus, r=r),
     partial(t6.g_t, r=r, nu=nu),
     partial(t6.g_huber2, r=r, qH=qH)]

psi = [partial(t6.psi_gaus),
       partial(t6.psi_t, r=r, nu=nu),
       partial(t6.psi_huber2, r=r, qH=qH)]

# needed for plots
x = np.arange(-20, 20.001, .1)
y = np.arange(-20, 20.001, .1)
X, Y = np.meshgrid(x, y)

g_names = ["Gaussian", "t", "Huber"]

plt.figure()
t6.plot_scatter(data, K_true, r)

pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

for m in range(K_true):
    Z = multivariate_normal(mean=mu_true[:,m], cov=S_true[m,:,:])
    Z = Z.pdf(pos)
    Z = np.reshape(Z, X.shape)
    plt.contour(X, Y, Z)

plt.title("Model: True")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

#%%
for iModel in range(len(g)):
    # perform EM algorithm

    mu_est, S_est, t, R = t6.EM_RES(data[:,1:], K_true, g[iModel], psi[iModel])
    
    plt.figure()
    t6.plot_scatter(data, K_true, r)
    for m in range(K_true):
        Z = multivariate_normal(mean=mu_est[m], cov=S_est[m,:,:])
        Z = Z.pdf(pos)
        Z = np.reshape(Z, X.shape)
        plt.contour(X, Y, Z)
        
    plt.title("Model: " + g_names[iModel])
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")




