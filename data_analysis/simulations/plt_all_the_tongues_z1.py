#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:54:39 2017

@author: burt
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
p = '/home/burt/neurospora/figures/entrainment/arrays/'


t12 = np.load(p+"2017_10_19_tongue_1_2_z_1.0.npz")
t21 = np.load(p+"2017_10_16_tongue_2_1_z_1.0.npz")
t13 = np.load(p+"2017_10_17_tongue_1_3_z_1.0.npz")
t11 = np.load(p+"2017_10_19_tongue_1_1_z_1.0.npz")

t12_wm = t12['warm_mesh']
t21_wm = t21['warm_mesh']
t13_wm = t13['warm_mesh']
t11_wm = t11['warm_mesh']

t12_zm = t12['zeit_mesh']
t21_zm = t21['zeit_mesh']
t13_zm = t13['zeit_mesh']
t11_zm = t11['zeit_mesh']

t12_ent = t12['ent']
t21_ent = t21['ent']
t13_ent = t13['ent']
t11_ent = t11['ent']

t12_ent[t12_ent != 2*np.pi] = 0.05 #blau
t21_ent[t21_ent != 2*np.pi] = 0.15 #orange
t13_ent[t13_ent != 2*np.pi] = 0.25 #grün
t11_ent[t11_ent != 2*np.pi] = 0.75 # grau

t12_mask = np.ma.masked_array(t12_ent, t12_ent == 2*np.pi)
t21_mask = np.ma.masked_array(t21_ent, t21_ent == 2*np.pi)
t13_mask = np.ma.masked_array(t13_ent, t13_ent == 2*np.pi)
t11_mask = np.ma.masked_array(t11_ent, t11_ent == 2*np.pi)



fig,ax = plt.subplots(figsize=(12,9))

ax.pcolormesh(t11_wm,t11_zm, t11_mask,cmap = "tab10", edgecolors = "none", vmin = 0, vmax = 1, label = "1")
ax.pcolormesh(t12_wm,t12_zm, t12_mask,cmap = "tab10", edgecolors = "none", vmin = 0, vmax = 1, label = "1")
ax.pcolormesh(t13_wm,t13_zm, t13_mask,cmap = "tab10", edgecolors = "none", vmin = 0, vmax = 1, label = "1")
ax.pcolormesh(t21_wm,t21_zm, t21_mask,cmap = "tab10", edgecolors = "none", vmin = 0, vmax = 1, label = "1")

grey_patch = mpatches.Patch(color='tab:grey', label='1:1')
blue_patch = mpatches.Patch(color='tab:blue', label='1:2')
green_patch = mpatches.Patch(color='tab:green', label='1:3')
orange_patch = mpatches.Patch(color='tab:orange', label='2:1')

ax.legend(handles=[grey_patch, blue_patch, green_patch, orange_patch], fontsize = 'xx-large')

ax.set_xticks([0,10,20,30,40,50,60])
ax.set_xlabel("T [h]", fontsize = "xx-large")
ax.set_ylabel("Z [a.u.]", fontsize = "xx-large")
#ax.set_yticklabels(["0","0.2","0.4","0.6","0.8","0.1"])
ax.tick_params(labelsize = "x-large")


plt.tight_layout()
fig.savefig("all_tongues_z1.pdf", dpi = 1200)
plt.show() 

