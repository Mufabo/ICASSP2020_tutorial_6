# -*- coding: utf-8 -*-
import scipy.special as sps
igamma = lambda a, b: sps.gammaincc(a, b)* sps.gamma(a)

import numpy as np
mldivide = lambda A, B: np.linalg.lstsq(B.conj().T, A.conj().T, rcond=None)[0].conj().T
