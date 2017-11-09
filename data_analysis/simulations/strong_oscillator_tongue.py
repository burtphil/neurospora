#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:16:27 2017

@author: burt
"""

import numpy as np
import matplotlib.pyplot as plt

x= np.linspace(-5,5,1000)

plt.plot(x,np.abs(x), "k--", label = "Weak oscillator")
plt.plot(x,2*np.abs(x),"k", label = "Strong oscillator")
plt.ylim(0,1)
plt.xlim(-1,1)
plt.tick_params(
    axis = "both",         
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',
    left = 'off',         # ticks along the top edge are off
    labelbottom='off',
    labelleft ='off')
plt.xlabel("T [a.u.]", fontsize= 'xx-large')
plt.ylabel("Z [a.u.]", fontsize = 'xx-large')
plt.legend(loc = 4)