"""

"""
from scipy.stats import multivariate_normal
import numpy as np
from scipy.stats.distributions import chi2
from scipy.special import gamma
from scipy.special import gammaincc
import matplotlib.pyplot as plt
from functools import partial
import icassp20_T6 as t6
import warnings

warnings.filterwarnings('ignore')

#%% User input
# Select combination of EM and BIC to be simulated
# 1: Gaussian, 2: t, 3: Huber, 4: Tukey
em_bic = np.array([[1, 1],
                   [2, 2],
                   [2, 4],
                   [3, 3],
                   [3, 4]])

# t
nu = 3
# Huber
qH = .8
# Tukey
cT = 4.685

#%% Your data here
epsilon = .15 # percentage of replacement outliers
N_k = 250 # Number of samples per cluster
data, r, N, K_true, mu_true, S_true = t6.data_31(N_k, epsilon)
import scipy.io as sio
data = sio.loadmat('C:/Users/Computer/projects/ICASSP2020_tutorial/tests/data_simple.mat')['data']
L_max = 2 * K_true # search range

#%% Model definitions
# Huber parameters
igamma = lambda a, b: gammaincc(a, b)* gamma(a)

cH = np.sqrt(chi2.ppf(qH, r))
bH = chi2.cdf(cH**2, r+2) + cH**2 / r * (1 - chi2.cdf(cH**2, r))
aH = gamma(r/2) / np.pi**(r/2) / ( (2*bH)**(r/2) * (gamma(r/2) - igamma(r/2, cH**2 / (2*bH))) + (2*bH*cH**r*np.exp(-cH**2/(2*bH))) / (cH**2 - bH*r))


# Density definitions

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

embic_iter = len(em_bic)
S_est = [[] for _ in range(L_max)]
mu_est = [[] for _ in range(L_max)]

bic = np.zeros([embic_iter, L_max, 3])
like = np.zeros([embic_iter, L_max, 3])
pen = np.zeros([embic_iter, L_max, 3])

#%% EM
for ii_embic in range(embic_iter):
    for ll in range(L_max):
        #EM
        mu, S, t, R = t6.EM_RES(data[:,1:], ll+1, g[em_bic[ii_embic, 0]-1], psi[em_bic[ii_embic,0]-1])
        
        mu_est[ll].append(mu)
        S_est[ll].append(S)
        
        mem = (R == R.max(axis=1)[:,None])
        
        #BIC        
        bic[ii_embic, ll, 0], like[ii_embic, ll, 0], pen[ii_embic, ll, 0] = t6.BIC_RES_2(data, S_est[ll][ii_embic], mu_est[ll][ii_embic], t, mem,rho[em_bic[ii_embic, 1]-1], psi[em_bic[ii_embic, 1]-1], eta[em_bic[ii_embic, 1]-1])            
        bic[ii_embic, ll, 1], like[ii_embic, ll, 1], pen[ii_embic, ll, 1] = t6.BIC_RES_asymptotic(S_est[ll][ii_embic], t, mem, rho[em_bic[ii_embic, 1]-1], psi[em_bic[ii_embic, 1]-1], eta[em_bic[ii_embic, 1]-1])
        bic[ii_embic, ll, 2], like[ii_embic, ll, 2], pen[ii_embic, ll, 2] = t6.BIC_S(S_est[ll][ii_embic], t, mem, rho[em_bic[ii_embic, 1]-1])

#%% Plots
x = np.arange(-20, 20.001, .1)
y = np.arange(-20, 20.001, .1)
X, Y = np.meshgrid(x, y)

g_names = ["Gaussian", "t", "Huber"]

marker = ['o','s','d','*','x','^','v','>','<','p','h', '+','o']
names = ["Finite", "Asymptotic", "Schwarz"]
g_names = ["Gaus", "t", "Huber", "Tukey"]

pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

#BIC
for ii_embic in range(embic_iter):
    plt.figure()
    plt.subplot(1,2,1)
    t6.plot_scatter(data, K_true, r)
    
    for m in range(K_true):
        mu = mu_est[K_true][ii_embic][m]
        S = S_est[K_true][ii_embic][m]
        Z = multivariate_normal.pdf(pos, mean=mu, cov=S)
        Z = np.reshape(Z, X.shape)
        plt.contour(X, Y, Z)
    plt.title("EM: " + g_names[em_bic[ii_embic,0]-1] +" at K = " + str(K_true))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.subplot(1,2,2)
    plt.plot(bic[ii_embic]) #, marker=marker[:bic.shape[-1]]
    plt.grid()
    plt.xlabel("number of clusters")
    plt.ylabel("BIC")
    plt.legend(names, loc="lower left")
    plt.title("Nk: " + str(N_k) + ", eps: "+str(epsilon) + ", EM-"+g_names[em_bic[ii_embic,0]-1] + ", BIC-"+g_names[em_bic[ii_embic,1]-1])    
    