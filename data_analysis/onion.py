# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:53:22 2017

@author: burt
"""

list = []


def func(warm, zeit):    
    crit = warm + zeit
    if crit < 25:
        return [warm, zeit, crit, True]
    else:
        return [warm, zeit, crit, False]

for i in range(10):
    zeitgeber_period = 18 + i
    for j in range((zeitgeber_period + 1)):
        warm_dur = j
        list.append(func(warm_dur, zeitgeber_period))
