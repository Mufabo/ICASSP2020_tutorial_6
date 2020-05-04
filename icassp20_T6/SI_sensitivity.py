"""
This file simulates the sensitivity curves.
"""

import numpy as np
import random
import icassp20_T6 as t6
from functools import partial
from scipy.stats.distributions import chi2
from scipy.special import gamma
from scipy.special import gammaincc
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt

#%% User input

# Number of samples per cluster
N_k = 50

# Nonte Carlo iterations
MC = 5

# range of outliers
out_range = np.array([[-20, 20], [-20, 20]])

# steps between outliers
step_eps = 10

# Select combinations of EM and BIC to be simulated
# 1: Gaussian, 2: t, 3: Huber, 4: Tukey
em_bic = np.array([[1, 1],
                   [2, 2],
                   [2, 4],
                   [3, 3],
                   [3, 4]])

# t
nu = 3

# Huber
qant = .8

# Tukey
cT = 4.685

#%% Data generation
# grid for single outlier
x = np.arange(out_range[0, 0], out_range[0, 1]+1, step_eps)
y = np.arange(out_range[1, 0], out_range[1, 1]+1, step_eps)
X, Y = np.meshgrid(x, y)

embic_iter = len(em_bic)
eps_iter = X.shape[0] * X.shape[1]

data = np.zeros([MC, eps_iter, N_k*3, 3])

for ii_eps in range(eps_iter):
    for ii_mc in range(MC):
        data[ii_mc, ii_eps, :, :], r, N, K_true, mu_true, S_true = t6.data_31(N_k, 0)
        
        # replacement outlier
        N_repl = 1
        index_repl = random.sample(range(N), N_repl)
        data[ii_mc, ii_eps, index_repl,:] = np.array([np.ones(N_repl) * (K_true+1), 
                                                      X.flatten()[ii_eps], Y.flatten()[ii_eps]])
        
L_max = 2 * K_true # search range

igamma = lambda a, b: gammaincc(a, b)* gamma(a)

cH = np.sqrt(chi2.ppf(qant, r))
bH = chi2.cdf(cH**2, r+2) + cH**2 / r * (1 - chi2.cdf(cH**2, r))
aH = gamma(r/2) / np.pi**(r/2) / ( (2*bH)**(r/2) * (gamma(r/2) - igamma(r/2, cH**2 / (2*bH))) + (2*bH*cH**r*np.exp(-cH**2/(2*bH))) / (cH**2 - bH*r))



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
    
    
tic = time.time()
pool = Pool()

breakpoint()
bic_final = np.zeros([embic_iter, 3, L_max, MC, eps_iter])
like_final = np.zeros([embic_iter, 3, L_max, MC, eps_iter])
pen_final = np.zeros([embic_iter, 3, L_max, MC, eps_iter])

def fun(iMC):
    bic = np.zeros([embic_iter, L_max, 3])
    like = np.zeros([embic_iter, L_max, 3])
    pen = np.zeros([embic_iter, L_max, 3])
    for ii_embic in range(embic_iter):
        for ll in range(L_max):
            # EM
            mu_est, S_est, t, R = t6.EM_RES(data[ii_eps], ll+1, g[em_bic[ii_embic, 0]-1], psi[em_bic[ii_embic,0]-1])
            mem = (R == R.max(axis=1)[:,None])
            
            bic[ii_embic, ll, 0], like[ii_embic, ll, 0], pen[ii_embic, ll, 0] = t6.BIC_RES_2(data, S_est, mu_est, t, mem,rho[em_bic[ii_embic, 1]-1], psi[em_bic[ii_embic, 1]-1], eta[em_bic[ii_embic, 1]-1])            
            bic[ii_embic, ll, 1], like[ii_embic, ll, 1], pen[ii_embic, ll, 1] = t6.BIC_RES_asymptotic(S_est, t, mem, rho[em_bic[ii_embic, 1]-1], psi[em_bic[ii_embic, 1]-1], eta[em_bic[ii_embic, 1]-1])
            bic[ii_embic, ll, 2], like[ii_embic, ll, 2], pen[ii_embic, ll, 2] = t6.BIC_S(S_est, t, mem, rho[em_bic[ii_embic, 1]-1])
            
            for ii_embic in range(embic_iter):
                for ll in range(L_max):
                    # EM
                    mu_est, S_est, t, R = t6.EM_RES(data[ii_mc, ii_eps, :, 1:r+1], ll+1, g[em_bic[ii_embic,0]], psi[em_bic[ii_embic,0]])
                    mem = (R == R.max(axis=1)[:,None])
                    
                    # BIC
                    bic[ii_embic, ll, 0], like[ii_embic, ll, 0], pen[ii_embic, ll, 0] = t6.BIC_RES_2(data[iMC, ii_eps, :, :], S_est, mu_est, t, mem,rho[em_bic[ii_embic, 1]-1], psi[em_bic[ii_embic, 1]-1], eta[em_bic[ii_embic, 1]-1])            
            
                    bic[ii_embic, ll, 1], like[ii_embic, ll, 0], pen[ii_embic, ll, 0] = t6.BIC_RES_asymptotic(S_est, t, mem, rho[em_bic[ii_embic, 1]-1], psi[em_bic[ii_embic, 1]-1], eta[em_bic[ii_embic, 1]-1])
            
                    bic[ii_embic, ll, 2], like[ii_embic, ll, 0], pen[ii_embic, ll, 0] = t6.BIC_S(S_est, t, mem, rho[em_bic[ii_embic, 1]-1])
                    
            bic_final[:,:,:, iMC, ii_eps] = bic
            like_final[:,:,:, iMC, ii_eps] = like
            pen_final[:,:,:, iMC, ii_eps] = pen
            print(str(ii_eps))
            print(time.time() - tic)
            
for ii_eps in range(eps_iter):
    pool.map(fun, [iMC for iMC in range(MC)])


p_under = np.zeros([embic_iter, bic_final.shape[1], eps_iter])
p_det = np.zeros([embic_iter, bic_final.shape[1], eps_iter])
p_over = np.zeros([embic_iter, bic_final.shape[1], eps_iter])
       
#%% Evaluation
for ii_embic in range(embic_iter):
    for ii_eps in range(eps_iter):
        for k in range(bic_final.shape[0]):
            
            BICmax = bic_final[k,:, ii_embic, :, ii_eps] == np.max(bic_final[k,:, ii_embic, :, ii_eps], axis=0)
            
            K_true_det = np.repeat(np.hstack([[K_true == s for s in range(1, K_true+1)], np.zeros(L_max - K_true)]), MC) == 1

            K_true_det = np.reshape(K_true_det, [L_max, MC])

            K_true_under = np.repeat(np.hstack([np.invert(\
               np.array([K_true == s for s in range(1, K_true)])),\
               np.zeros(L_max - (K_true-1))]) , MC) == 1
                
            K_true_under = np.reshape(K_true_under, [L_max, MC])
            
            p_under[ii_embic, k, ii_eps] = np.sum(BICmax[K_true_under.T])/MC
            p_det[ii_embic, k, ii_eps] = np.sum(BICmax[K_true_det.T])/MC
            p_over[ii_embic, k, ii_eps] = 1 - p_det[ii_embic, k, ii_eps] - p_under[ii_embic, k, ii_eps]
            

#%% Plot
g_names = ["Gaus", "t", "Huber", "Tukey"]
names = ["RES", "aRES", "Schwarz"]
p_det_2 = np.transpose(p_det, [0, 2, 1])
data, r, N, K_true, mu_true, S_true = t6.data_31(N_k, 0)

for ii_embic in range(embic_iter):
    for k_bic in range(bic_final.shape[1]):
        Z = np.reshape(p_det_2[ii_embic, :, k_bic], X.shape)
        plt.figure()
        M, c = plt.contour(X, Y, Z)
        t6.plot_scatter(data, K_true, r)
        plt.title("EM-" + g_names[em_bic[ii_embic, 0]-1] + ", BIC-" + g_names[em_bic[ii_embic, 1]-1] + "-"+names[k_bic])
        