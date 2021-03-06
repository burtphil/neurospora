#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:53:10 2017

@author: burt
"""

import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(10,20,20)
y = np.linspace(200,220,20)

X,Y = np.meshgrid(x,y)

def func(a,b):
    return (a+b)

plt.pcolormesh(X,Y,func(X,Y))
plt.show()

Z = np.zeros_like(X)

list = []


def loop(warm, zeit):    
    crit = warm + zeit
    if crit < 25:
        return [True]
    else:
        return [False]

for i in x:
    for j in y:
        list.append(loop(i, j))

### make a for loop with enumerate
ints = [8, 23, 45, 12, 78]

for idx, valx in enumerate(x):
    for idy, valy in enumerate(y):
        Z[idx,idy] = valx + valy