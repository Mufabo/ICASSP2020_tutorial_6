# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:04:16 2020

@author: Computer
"""

import scipy.io as sio
import numpy as np

bic_final = sio.loadmat('C:/Users/Computer/projects/ICASSP2020_tutorial/tests/bic_final.mat')['bic_final']
bic_final = np.transpose(bic_final, [2,3,4,0,1])

embic_iter = 5
eps_iter = 25

for ii_embic in range(embic_iter):
    for ii_eps in range(eps_iter):
        for k in range(bic_final.shape[1]):
            BICmax = bic_final[:, k, ii_embic, :, ii_eps] == np.max(bic_final[:, k, ii_embic, :, ii_eps], axis=0)
