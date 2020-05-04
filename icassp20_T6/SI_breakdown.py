"""
This file simulates the probability of detection over the percentage of 
outliers
"""
import numpy as np
from scipy.stats.distributions import chi2
from scipy.special import gamma
from scipy.special import gammaincc
import matplotlib.pyplot as plt
from functools import partial
import icassp20_T6 as t6
import time
from multiprocessing import Pool

#%% User input

# percentage of replacement outliers
epsilon = np.arange(0,.351,.05)

# Number of data points per cluster
N_k = 10

# Monte Carlo iterations
MC = 30

# Select combination of EM and BIC to be simulated
# 1: Gaussian, 2: t, 3: Huber, 4: Tukey
em_bic = np.array([[1, 1],
                   [2, 2],
                   [2, 4],
                   [3, 3],
                   [3, 4]])

# data generation
embic_iter = len(em_bic)
eps_iter = len(epsilon)
data = np.zeros([MC, eps_iter, 3*N_k ,3])


for iEpsilon in range(eps_iter):
    for iMC in range(MC):
        data[iMC, iEpsilon, :,:], r, N, K_true, mu_true, S_true = t6.data_31(N_k, epsilon[iEpsilon])
        

L_max = 2*K_true # search range

# Tukey:
cT = 4.685

# t:
nu = 3

# Huber:
qant = 0.8

igamma = lambda a, b: gammaincc(a, b)* gamma(a)

cH = np.sqrt(chi2.ppf(qant, r))
bH = chi2.cdf(cH**2, r+2) + cH**2 / r * (1 - chi2.cdf(cH**2, r))
aH = gamma(r/2) / np.pi**(r/2) / ( (2*bH)**(r/2) * (gamma(r/2) - igamma(r/2, cH**2 / (2*bH))) + (2*bH*cH**r*np.exp(-cH**2/(2*bH))) / (cH**2 - bH*r))


#%% Density definitions

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

bic_final = np.zeros([embic_iter, L_max, 3, MC, eps_iter])
like_final = np.zeros([embic_iter, L_max, 3, MC, eps_iter])
pen_final = np.zeros([embic_iter, L_max, 3, MC, eps_iter])

#%% Cluster Enumeration
tic = time.time()
pool = Pool()

def fun(iMC):
    bic = np.zeros([embic_iter, L_max, 3])
    like = np.zeros([embic_iter, L_max, 3])
    pen = np.zeros([embic_iter, L_max, 3])
    for iEmBic in range(embic_iter):
        for ll in range(L_max):
            # EM
            mu_est, S_est, t, R = t6.EM_RES(data[:,1:r+1,iEpsilon,iMC], ll+1, g[em_bic[iEmBic, 0]-1], psi[em_bic[iEmBic,0]-1])
            mem = (R == R.max(axis=1)[:,None])
            
            bic[iEmBic, ll, 0], like[iEmBic, ll, 0], pen[iEmBic, ll, 0] = t6.BIC_RES_2(data, S_est, mu_est, t, mem,rho[em_bic[iEmBic, 1]-1], psi[em_bic[iEmBic, 1]-1], eta[em_bic[iEmBic, 1]-1])            
            bic[iEmBic, ll, 1], like[iEmBic, ll, 1], pen[iEmBic, ll, 1] = t6.BIC_RES_asymptotic(S_est, t, mem, rho[em_bic[iEmBic, 1]-1], psi[em_bic[iEmBic, 1]-1], eta[em_bic[iEmBic, 1]-1])
            bic[iEmBic, ll, 2], like[iEmBic, ll, 2], pen[iEmBic, ll, 2] = t6.BIC_S(S_est, t, mem, rho[em_bic[iEmBic, 1]-1])
            
            for iEmBic in range(embic_iter):
                for ll in range(L_max):
                    # EM
                    mu_est, S_est, t, R = t6.EM_RES(data[:,1:r+2, iEpsilon, iMC], ll+1, g[em_bic[iEmBic,0]], psi[em_bic[iEmBic,0]])
                    mem = (R == R.max(axis=1)[:,None])
                    
                    # BIC
                    bic[iEmBic, ll, 0], like[iEmBic, ll, 0], pen[iEmBic, ll, 0] = t6.BIC_RES_2(data[iMC, iEpsilon, :, :], S_est, mu_est, t, mem,rho[em_bic[iEmBic, 1]-1], psi[em_bic[iEmBic, 1]-1], eta[em_bic[iEmBic, 1]-1])            
            
                    bic[iEmBic, ll, 1], like[iEmBic, ll, 0], pen[iEmBic, ll, 0] = t6.BIC_RES_asymptotic(S_est, t, mem, rho[em_bic[iEmBic, 1]-1], psi[em_bic[iEmBic, 1]-1], eta[em_bic[iEmBic, 1]-1])
            
                    bic[iEmBic, ll, 2], like[iEmBic, ll, 0], pen[iEmBic, ll, 0] = t6.BIC_S(S_est, t, mem, rho[em_bic[iEmBic, 1]-1])
                    
            bic_final[:,:,:, iMC, iEpsilon] = bic
            like_final[:,:,:, iMC, iEpsilon] = like
            pen_final[:,:,:, iMC, iEpsilon] = pen
            print(epsilon[iEpsilon])
            print(time.time() - tic)


            
for iEpsilon in range(eps_iter):
    #pool.imap(fun, [iMC for iMC in range(MC)])
    for iMC in range(MC):
        bic = np.zeros([embic_iter, L_max, 3])
        like = np.zeros([embic_iter, L_max, 3])
        pen = np.zeros([embic_iter, L_max, 3])
        for iEmBic in range(embic_iter):
            for ll in range(L_max):
                # EM
                mu_est, S_est, t, R = t6.EM_RES(data[iMC, iEpsilon,  :, 1:r+1], ll+1, g[em_bic[iEmBic, 0]-1], psi[em_bic[iEmBic,0]-1])
                mem = (R == R.max(axis=1)[:,None])
                
                bic[iEmBic, ll, 0], like[iEmBic, ll, 0], pen[iEmBic, ll, 0] = t6.BIC_RES_2(data[iMC, iEpsilon, :, :], S_est, mu_est, t, mem,rho[em_bic[iEmBic, 1]-1], psi[em_bic[iEmBic, 1]-1], eta[em_bic[iEmBic, 1]-1])            
                bic[iEmBic, ll, 1], like[iEmBic, ll, 1], pen[iEmBic, ll, 1] = t6.BIC_RES_asymptotic(S_est, t, mem, rho[em_bic[iEmBic, 1]-1], psi[em_bic[iEmBic, 1]-1], eta[em_bic[iEmBic, 1]-1])
                bic[iEmBic, ll, 2], like[iEmBic, ll, 2], pen[iEmBic, ll, 2] = t6.BIC_S(S_est, t, mem, rho[em_bic[iEmBic, 1]-1])
                   
        bic_final[:,:,:, iMC, iEpsilon] = bic
        like_final[:,:,:, iMC, iEpsilon] = like
        pen_final[:,:,:, iMC, iEpsilon] = pen
        print(epsilon[iEpsilon])
        print(time.time() - tic)

p_under = np.zeros([embic_iter, L_max, eps_iter])
p_det = np.zeros([embic_iter, L_max, eps_iter])
p_over = np.zeros([embic_iter, L_max, eps_iter])

#%%
# Evaluation
for iEmBic in range(embic_iter):
    for iEpsilon in range(eps_iter):
        for k in range(3):
            BICmax = bic_final[iEmBic, :,k , :, iEpsilon] == np.max(bic_final[iEmBic, :,k , :, iEpsilon], axis=0)

            K_true_det = np.repeat(np.hstack([[K_true == s for s in range(1, K_true+1)], np.zeros(L_max - K_true)]), MC) == 1
            K_true_det = np.reshape(K_true_det, [L_max, MC])
            
            K_true_under = np.repeat(np.hstack([np.invert(\
               np.array([K_true == s for s in range(1, K_true)])),\
               np.zeros(L_max - (K_true-1))]) , MC) == 1
            K_true_under = np.reshape(K_true_under, [L_max, MC])

            p_under[iEmBic, k, iEpsilon] = np.sum(BICmax[K_true_under])/MC
            p_det[iEmBic, k, iEpsilon] = np.sum(BICmax[K_true_det])/MC
            p_over[iEmBic, k, iEpsilon] = 1 - p_det[iEmBic, k, iEpsilon] - p_under[iEmBic, k, iEpsilon]
            

#%% Plots

g_names = ['Gaus', 't', 'Huber', 'Tukey']
marker = ['o','s','d','*','x','^','v','>','<','p','h', '+','o']
names = ["RES", "aRES", "Schwarz"]

for iEmBic in range(embic_iter):
    plt.figure()
    plt.grid()
    plt.plot(epsilon, p_det[iEmBic, :, :])
    plt.xlabel('% of Outliers')
    plt.ylabel("probability of detection")
    plt.ylim(0, 1)
    plt.legend(names, loc='lower left')
    plt.title("Nk-" + str(N_k) + ", EM-" + g_names[em_bic[iEmBic, 0]-1] 
              + ", BIC-" + g_names[em_bic[iEmBic, 1]-1])
    
names_all = ["EM-" + g_names[em_bic[iEmBic, 0]-1] 
              + ", BIC-" + g_names[em_bic[iEmBic, 1]-1] for iEmBic in range(embic_iter)]

plt.figure()
plt.grid()
plt.xlabel("% of Outliers")
plt.ylabel("Probability of detection")
plt.ylim(0, 1)

for iEmBic in range(embic_iter):
    plt.plot(epsilon, p_det[iEmBic, :, :], marker=marker[iEmBic])
    
plt.legend(names_all, loc="lower right")
plt.title("Nk-" + str(N_k))

for ii_bic in range(bic_final.shape[1]):
    plt.figure()
    plt.grid()
    plt.plot(epsilon, p_det[:, ii_bic, :], marker=marker[ii_bic])
    plt.xlabel("% of outliers")
    plt.ylabel("Probability of detection")
    plt.ylim(0, 1)
    plt.legend(["EM: " + g_names[em_bic[iEmBic, 0]-1] + ", BIC" + g_names[em_bic[iEmBic, 1]-1]], loc="lower left")
    plt.title("Nk-" + str(N_k) + "BIC-" + names[ii_bic])
    
    
            
