#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:04:54 2017

@author: burt
"""

import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc

k1 = np.load("k1_tongue_200_res.npz")

fig, ax = plt.subplots(figsize = (12,9))
heatmap = ax.pcolormesh(k1['warm_mesh'],k1['zeit_mesh'], k1['ent'], cmap = cc.m_fire, edgecolors = "none", vmin = 0, vmax = 2*np.pi)
cbar = fig.colorbar(heatmap,ticks=[0,np.pi/2,np.pi,1.5*np.pi,2*np.pi], label = 'Phase [rad]')
cbar.ax.set_yticklabels(['0','$\pi/2$','$\pi$','$3\pi/2$', '2$\pi$'])
plt.xlabel("T [h]", fontsize = 22)
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
plt.ylabel("Z [a.u.]", fontsize = 22)
plt.tight_layout()
fig.savefig("tongue_1_1.pdf", dpi = 1200)
plt.show()  