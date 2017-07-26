#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:58:54 2017

@author: burt
"""

import pandas as pd
import numpy as np

nan = float('nan')

array = np.array([nan,nan,nan])

barray = np.array([1, nan, 0])
c = np.empty_like(barray)
carray = np.array([1,0])

print array.any()
print barray.any()
print carray.any()

a = np.array([1,4])
b = np.array([0,1,2])

print a + b