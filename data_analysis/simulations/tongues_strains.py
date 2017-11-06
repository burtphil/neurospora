#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:54:39 2017

@author: burt
"""

import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
p = '/home/burt/neurospora/figures/entrainment/arrays/'


frq = np.load(p+"2017_10_16_tongue_1_1_z_0.1_ph.npz")
frq1 = np.load(p+'2017_10_24_frq1_tongue_1_1_z_0.1_ph.npz')
frq7 = np.load(p+"2017_10_25_frq7_tongue_1_1_z_0.1_ph.npz")


frq_wm = frq['warm_mesh']
frq1_wm = frq1['warm_mesh']
frq7_wm = frq7['warm_mesh']



frq_zm = frq['zeit_mesh']
frq1_zm = frq1['zeit_mesh']
frq7_zm = frq7['zeit_mesh']



frq_ent = frq['ent']
frq1_ent = frq1['ent']
frq7_ent = frq7['ent']


labels = ["frq1","frq+","frq7"]
ticks = [16,22,28]


frq_mask = np.ma.masked_array(frq_ent, frq_ent == 2*np.pi)
frq1_mask = np.ma.masked_array(frq1_ent, frq1_ent == 2*np.pi)
frq7_mask = np.ma.masked_array(frq7_ent, frq7_ent == 2*np.pi)


fig,ax = plt.subplots(figsize=(12,9))



ax.pcolormesh(frq1_wm,frq1_zm, frq1_ent,cmap = cc.m_fire, edgecolors = "none")
ax.pcolormesh(frq7_wm,frq7_zm, frq7_ent,cmap = cc.m_fire, edgecolors = "none")
heatmap = ax.pcolormesh(frq_wm,frq_zm, frq_mask,cmap = cc.m_fire, edgecolors = "none")

cbar = fig.colorbar(heatmap,ticks=[0,np.pi/2,np.pi,1.5*np.pi,2*np.pi], label = 'Phase [rad]')
cbar.ax.set_yticklabels(['0','$\pi/2$','$\pi$','$3\pi/2$', '2$\pi$'])
cbar.ax.tick_params(labelsize = 16)
cbar.ax.set_ylabel("Phase [rad]", fontsize = 22)

ax.set_xticks([0,10,20,30,40])
ax.set_xlabel("T [h]", fontsize = 22)
ax.set_ylabel("Z [a.u.]", fontsize = 22)
ax.set_yticklabels(["0","0.02","0.04","0.06","0.08","0.1"])
ax.tick_params(labelsize = 16)

ax2 = ax.twiny()
ax2.set_xlim(0,40)
ax2.set_xticks(ticks)
ax2.set_xticklabels(labels)
ax2.tick_params(length = 0, labelsize = 16)

plt.tight_layout()
fig.savefig("tongues_strains.pdf", dpi = 1200)
plt.show() 


