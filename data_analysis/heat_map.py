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