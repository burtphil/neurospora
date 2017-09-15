# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:24:44 2017

@author: Philipp
"""

import numpy as np
import matplotlib.pyplot as plt

y = np.linspace(0,10,100)
x = np.linspace(0,10,100)

gridx, gridy = np.meshgrid(x,y)

entr = np.zeros_like(gridx)

for idx, valx in enumerate(x):
    for idy, valy in enumerate(y):
        entr[idy,idx] = valx*valy

ent = entr[:-1,:-1]

plt.pcolormesh(gridx,gridy, ent, cmap = "Blues", edgecolors = "none")
plt.colorbar()
plt.show()

fig, ax = plt.subplots()
heatmap = ax.pcolormesh(gridx,gridy, ent, cmap = "Blues", edgecolors = "none")
#legend

cbar = fig.colorbar(heatmap,ticks=[0,100], label = "PHASE")
cbar.ax.set_yticklabels(['0', '2pi'])
plt.show()