#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:54:39 2017

@author: burt
"""

import numpy as np
import matplotlib.pyplot as plt

p = '/home/burt/neurospora/figures/entrainment/tongue_'

t12 = np.load(p+"1_2.npz")
t21 = np.load(p+"2_1.npz")
t31 = np.load(p+"3_1.npz")
t13 = np.load(p+"1_3.npz")
t32 = np.load(p+"3_2.npz")
t23 = np.load(p+"2_3.npz")
t11 = np.load(p+"1_1.npz")

t12_wm = t12['warm_mesh']
t21_wm = t21['warm_mesh']
t31_wm = t31['warm_mesh']
t13_wm = t13['warm_mesh']
t32_wm = t32['warm_mesh']
t23_wm = t23['warm_mesh']
t11_wm = t11['warm_mesh']


t12_zm = t12['zeit_mesh']
t21_zm = t21['zeit_mesh']
t31_zm = t31['zeit_mesh']
t13_zm = t13['zeit_mesh']
t32_zm = t32['zeit_mesh']
t23_zm = t23['zeit_mesh']
t11_zm = t11['zeit_mesh']


t12_ent = t12['ent']
t21_ent = t21['ent']
t31_ent = t31['ent']
t13_ent = t13['ent']
t32_ent = t32['ent']
t23_ent = t23['ent']
t11_ent = t11['ent']

labels = ["1:3","1:2","2:3","1:1","3:2","2:1","3:1"]
ticks = [7.3,11,14.6,22,33,44,66]

fig = plt.figure(figsize=(12,9))
plt.pcolormesh(t11_wm,t11_zm, t11_ent,cmap = "Greys_r", edgecolors = "none")
plt.pcolormesh(t12_wm,t12_zm, t12_ent,cmap = "Greys_r", edgecolors = "none")
plt.pcolormesh(t13_wm,t13_zm, t13_ent,cmap = "Greys_r", edgecolors = "none")
plt.pcolormesh(t21_wm,t21_zm, t21_ent,cmap = "Greys_r", edgecolors = "none")
plt.pcolormesh(t31_wm,t31_zm, t31_ent,cmap = "Greys_r", edgecolors = "none")
plt.pcolormesh(t32_wm,t32_zm, t32_ent,cmap = "Greys_r", edgecolors = "none")
plt.pcolormesh(t23_wm,t23_zm, t23_ent,cmap = "Greys_r", edgecolors = "none")
plt.xticks(ticks, labels, fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlabel("T [h]", fontsize = 22)
plt.ylabel("Z [a.u.]", fontsize = 22)
plt.tight_layout()
fig.savefig("all_tongues.png", dpi = 1200)
plt.show() 