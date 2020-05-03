# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:21:53 2020

@author: Computer
"""

import numpy as np
from itertools import combinations
A = (np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]]).T - np.ones(3)[:,None]).T
b = np.array([[.1,.2,.3], [.2,.1,.4], [.3,.4,.5]])

num_vars = A.shape[1]
rank = np.linalg.matrix_rank(A)
if rank == num_vars:              
    sol = np.linalg.lstsq(A, b)[0]    # not under-determined
else:
    for nz in combinations(range(num_vars), rank):    # the variables not set to zero
        try: 
            sol = np.zeros((num_vars, 1))  
            sol[nz, :] = np.asarray(np.linalg.lstsq(A[:, nz], b))
            print(sol)
        except np.linalg.LinAlgError:     
            pass                    # picked bad variables, can't solve
            
print(sol)

# Matlabs / aka mldivide
mldivide = lambda A, B: np.linalg.lstsq(B.conj().T, A.conj().T, rcond=None)[0].conj().T

