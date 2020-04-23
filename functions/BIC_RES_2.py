import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt

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

# Cluster Enumeration
tic = time.time()
embic_iter = len(em_bic)


