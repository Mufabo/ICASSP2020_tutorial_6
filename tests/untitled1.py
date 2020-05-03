# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:53:03 2020

@author: Computer
"""
from functools import partial

def d():
    return 5

g = partial(d)

print(g())