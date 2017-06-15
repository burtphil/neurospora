# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:20:46 2017

@author: Philipp
"""
import numpy as np
from scipy.optimize import curve_fit



### take values for k5 and K from supplements hong et al
X = np.array([293.15, 294.15, 295.15, 296.15, 297.15, 298.15, 299.15, 300.15, 301.15, 302.15, 303.15])
y_k5 = np.array([0.238, 0.245, 0.25, 0.259, 0.266, 0.27, 0.275, 0.278, 0.28, 0.282, 0.285])
y_K  = np.array([0.001, 0.005, 0.02, 0.3, 1, 1.25, 2, 2.1, 2.4, 2.7, 3])

### define function to fit arrhenius incl temp dependency

def func(x,a,b):
    return a * x + b
            
k5_opt, k5_cov = curve_fit(func,X,y_k5)


K_opt, K_cov = curve_fit(func,X,y_K)

