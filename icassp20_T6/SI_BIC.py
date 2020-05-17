import numpy as np
from scipy.stats.distributions import chi2
from scipy.special import gamma
from scipy.special import gammaincc
import time
import matplotlib.pyplot as plt
from functools import partial
import icassp20_T6 as t6

# This file simulates the BIC, likelihood and penalty terms for a given
# data set.


"""
User input
"""

# number of Monte Carlo iterations
MC = 5

# percentage of replacement outliers
epsilon = 0.04

# Number of samples per cluster
N_k = 100

# Select combinations of EM and BIC to be simulated
# 1: Gaussian, 2: t, 3: Huber, 4: Tukey
em_bic = np.array([[1, 1],
                   [2, 2],
                   [2, 4],
                   [3, 3],
                   [3, 4]])


"""
Cluster Enumeration
"""

tic = time.time()
embic_iter = len(em_bic)

igamma = lambda a, b: gammaincc(a, b)* gamma(a)

data, labels, r, N, K_true, mu_true, S_true = t6.data_31(N_k, epsilon)
bic = np.zeros([embic_iter, 3, MC, 2*K_true])
like = np.zeros([embic_iter, 3, MC, 2*K_true])
pen = np.zeros([embic_iter, 3, MC, 2*K_true])

for iMC in range(MC):
    data, labels, r, N, K_true, mu_true, S_true = t6.data_31(N_k, epsilon)
    L_max = 2 * K_true # search range
    """
    Design parameters
    """
    
    # t:
    nu = 3
    
    # Huber:
    qant = 0.8
    
    cH = np.sqrt(chi2.ppf(qant, r))
    bH = chi2.cdf(cH**2, r+2) + cH**2 / r * (1 - chi2.cdf(cH**2, r))
    aH = gamma(r/2) / np.pi**(r/2) / ( (2*bH)**(r/2) * (gamma(r/2) - igamma(r/2, cH**2 / (2*bH))) + (2*bH*cH**r*np.exp(-cH**2/(2*bH))) / (cH**2 - bH*r))
    
    # Tukey:
    cT = 4.685
    
    """
    Density definitions
    """
    g = [partial(t6.g_gaus, r=r),
         partial(t6.g_t, r=r, nu=nu),
         partial(t6.g_huber2, r=r, cH=cH, bH=bH, aH=aH)]
    
    rho = [partial(t6.rho_gaus, r=r),
           partial(t6.rho_t, r=r, nu=nu),
           partial(t6.rho_huber2, r=r, cH=cH, bH=bH, aH=aH),
           partial(t6.rho_tukey, r=r, cT=cT)]
    
    psi = [partial(t6.psi_gaus),
           partial(t6.psi_t, r=r, nu=nu),
           partial(t6.psi_huber2, r=r, cH=cH, bH=bH),
           partial(t6.psi_tukey, cT=cT)]
    
    eta = [partial(t6.eta_gaus),
           partial(t6.eta_t, r=r, nu=nu),
           partial(t6.eta_huber2, r=r, cH=cH, bH=bH),
           partial(t6.eta_tukey, cT=cT)]
        
    for ii_embic in range(embic_iter):
        for ll in range(L_max):
            
            """
            EM
            """
            mu_est, S_est, t, R = t6.EM_RES(data, ll+1, g[em_bic[ii_embic, 0]-1], psi[em_bic[ii_embic,0]-1])
            mem = (R == R.max(axis=1)[:,None])            
            """
            BIC
            """

            bic[ii_embic, 0, iMC, ll], like[ii_embic, 0, iMC, ll], pen[ii_embic, 0, iMC, ll] = t6.BIC_F(data, S_est, mu_est, t, mem,rho[em_bic[ii_embic, 1]-1], psi[em_bic[ii_embic, 1]-1], eta[em_bic[ii_embic, 1]-1])            
            
            bic[ii_embic, 1, iMC, ll], like[ii_embic, 1, iMC, ll], pen[ii_embic, 1, iMC, ll] = t6.BIC_RES_asymptotic(S_est, t, mem, rho[em_bic[ii_embic, 1]-1], psi[em_bic[ii_embic, 1]-1], eta[em_bic[ii_embic, 1]-1])
            
            bic[ii_embic, 2, iMC, ll], like[ii_embic, 2, iMC, ll], pen[ii_embic, 2, iMC, ll] = t6.BIC_S(S_est, t, mem, rho[em_bic[ii_embic, 1]-1])
          
    print("Current iteration: %i" % iMC)
    
"""
Averaging over MC
"""
bic_avg = np.mean(bic, axis=2)
like_avg = np.mean(like, axis=2)
pen_avg = np.mean(pen, axis=2)
#%%  
"""
Plots
"""
t6.plot_scatter(data, K_true, r)

g_names = ['Gaus', 't', 'Huber', 'Tukey']
marker = ['o','s','d','*','x','^','v','>','<','p','h', '+','o']
names = ["RES", "aRES", "Schwarz"]

"""
BICs
"""
for ii_embic in range(embic_iter):
    plt.figure()
    plt.grid()
    [plt.plot(bic_avg[ii_embic,i,:], marker='o') for i in range(len(bic_avg[ii_embic]))]
    plt.xlabel('number of clusters')
    plt.ylabel('BIC')
    
    plt.legend(names, loc='lower right')
    plt.title("Nk: " + str(N_k) + ', eps: ' + str(epsilon) + " EM: " + g_names[em_bic[ii_embic, 0]-1] + ", BIC: " + g_names[em_bic[ii_embic, 1]-1])

"""
Likelihood
"""
plt.figure()
plt.grid()
leg_names = []
for ii_embic in range(embic_iter):
    [plt.plot(like_avg[ii_embic,i,:], marker='o') for i in range(len(like_avg[ii_embic]))]
    leg_names.append("EM: " + g_names[em_bic[ii_embic, 0]-1] + ", BIC: " + g_names[em_bic[ii_embic, 1]-1])

plt.xlabel('number of clusters')
plt.ylabel('Likelihood')

plt.legend(leg_names, loc='lower right')
plt.title("Nk: " + str(N_k) + ', eps: ' + str(epsilon))

"""
Penalty terms
"""
for ii_embic in range(embic_iter):
    plt.figure()
    plt.grid()
    [plt.plot(pen_avg[ii_embic,i,:], marker='o') for i in range(len(pen_avg[ii_embic]))]
    plt.xlabel('number of clusters')
    plt.ylabel('BIC')
    
    plt.legend(names, loc='lower right')
    plt.title("Nk: " + str(N_k) + ', eps: ' + str(epsilon) + " EM: " + g_names[em_bic[ii_embic, 0]-1] + ", BIC: " + g_names[em_bic[ii_embic, 1]-1])



    


