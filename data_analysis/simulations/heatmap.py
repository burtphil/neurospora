# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:24:44 2017

@author: Philipp
"""

import numpy as np
import matplotlib.pyplot as plt

y = np.linspace(0,2,10)
x = np.linspace(0,1,10)

gridx, gridy = np.meshgrid(x,y)

entr = np.zeros_like(gridx)

for idx, valx in enumerate(x):
    for idy, valy in enumerate(y):
        entr[idy,idx] = valx + valy

ent = entr[:-1,:-1]

plt.pcolormesh(gridx,gridy, ent, edgecolors = "k", vmin = 0, vmax = 1)
plt.colorbar()
plt.show()