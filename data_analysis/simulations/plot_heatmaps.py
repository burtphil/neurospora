#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:54:39 2017

@author: burt
"""

import numpy as np
import matplotlib.pyplot as plt

p = '/home/burt/neurospora/figures/entrainment/arrays/'


t12 = np.load(p+"2017_10_19_tongue_1_2_z_1.0.npz")
t11 = np.load(p+"2017_10_19_tongue_1_1_z_1.0.npz")
t21 = np.load(p+"2017_10_19_tongue_2_1_z_1.0.npz")

t12_wm = t12['warm_mesh']
t11_wm = t11['warm_mesh']
t21_wm = t21['warm_mesh']

t12_zm = t12['zeit_mesh']
t11_zm = t11['zeit_mesh']
t21_zm = t21['zeit_mesh']


t12_ent = t12['ent']
t11_ent = t11['ent']
t21_ent = t21['ent']


dummymin = np.zeros((100,1), dtype=np.float64)
dummymax = np.ones((100,1), dtype=np.float64)*60
wm =np.concatenate((dummymin, t12_wm,dummymax), axis = 1 )

dummyzm = np.linspace(0,1,100)*np.ones((100,1), dtype=np.float64)

zm =np.concatenate((dummyzm, dummyzm,t12_zm), axis = 1 )

test = np.vstack((t12_wm,t11_wm, t21_wm))
test2 = np.vstack((t12_zm,t11_zm, t21_zm))
ent = np.vstack((t12_ent,t11_ent,t21_ent))


fig,ax = plt.subplots(figsize=(12,9))

#ax.pcolormesh(t12_wm,t12_zm, t12_ent,cmap = "Greys_r", edgecolors = "none",vmin = 0, vmax = 2*np.pi)

#ax.pcolormesh(t11_wm,t11_zm, t11_ent,cmap = "Blues_r", edgecolors = "none",vmin = 0, vmax = 2*np.pi)

#ax.pcolormesh(t21_wm,t21_zm, t21_ent,cmap = "Reds_r", edgecolors = "none",vmin = 0, vmax = 2*np.pi)

ax.pcolormesh(test,test2, ent,cmap = "Reds_r", edgecolors = "none",vmin = 0, vmax = 2*np.pi)

plt.tight_layout()
plt.show() 


