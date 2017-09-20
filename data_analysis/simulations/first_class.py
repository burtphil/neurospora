#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:40:49 2017

@author: burt
"""
import numpy as np
import matplotlib.pyplot as plt

def get_extrema(y,t):
    """
    take two arrays: y values and corresponding time array
    finds local maxima and minima
    finds adjacent values next to local maxima and minima
    return list with maxima and minima
    both list entries contain three arrays corresponding to actual extrema, and both neighbors
    """
    assert y.size > 3, "y array passed to get_extrema not large enough"
    
    imax = y.size-1
    i = 1
    tmax = []
    tmax_after = []
    tmax_before = []  
    ymax = []
    ymax_after = []
    ymax_before = []
    
    tmin = []
    tmin_after = []
    tmin_before = [] 
    ymin = []
    ymin_after = []
    ymin_before = []

    while i < imax:
        
        if (y[i] > y[i+1]) & (y[i] > y[i-1]):
            tmax.append(t[i])
            tmax_after.append(t[i+1])
            tmax_before.append(t[i-1])
            ymax.append(y[i])
            ymax_after.append(y[i+1])
            ymax_before.append(y[i-1])

        if (y[i] < y[i+1]) & (y[i] < y[i-1]):
            tmin.append(t[i])
            tmin_after.append(t[i+1])
            tmin_before.append(t[i-1])
            ymin.append(y[i])
            ymin_after.append(y[i+1])
            ymin_before.append(y[i-1])
        i = i+1
    
    maxima = [tmax,tmax_before,tmax_after,ymax,ymax_before,ymax_after]
    maxima = np.array(maxima).T
    minima = [tmin,tmin_before,tmin_after,ymin,ymin_before,ymin_after]  
    minima = np.array(minima).T
    
    return([maxima,minima])

def interpolate(m):
    """
    takes an array with three x and three corresponding y values as input
    define parabolic function through three points and 
    returns local maximum as array([time, y value])     
    """
    
    x1 = m[0]
    x2 = m[1]
    x3 = m[2]
    y1 = m[3]
    y2 = m[4]
    y3 = m[5]
    denom = (x1 - x2)*(x1 - x3)*(x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    xext = -B/(2*A)
    yext = A*xext**2 + B*xext + C
    
    return(np.array([xext,yext]))

def get_max(a,t):
    ex = get_extrema(a,t)[0]
    out = ipol(ex,1)
    return out

def get_min(a,t):
    ex = get_extrema(a,t)[1]
    out = ipol(ex,1)
    return out

def get_xmax(a,t):
    ex = get_extrema(a,t)[0]
    out = ipol(ex,0)
    return out

def get_xmin(a,t):
    ex = get_extrema(a,t)[1]
    out = ipol(ex,0)
    return out

def ipol(arr,nr):
    l=[]
    for x in arr:
        l.append(interpolate(x)[nr])
    return np.asarray(l)

def get_periods(a,t):
    """
    take array containing time stamps of maxima
    returns the mean of time differences between maxima
    """
    ex = get_extrema(a,t)[1]
    
    l = ipol(ex,0)
    
    diff = np.diff(l)
    
    return diff


def get_phase(a,ta, b, tb):
    """
    takes two np arrays (containing maxima)
    choose first 10 maxima in both arrays
    subtract and check if entries are positive
    if not, subtract other way around
    return phase 2pi normalized over zeitgeber period
    """
    a = get_xmin(a,ta)
    b = get_xmin(b,tb)
    a = a[:10]
    b = b[:10]
    c = a-b
    if np.sum(c)>0:
        c=np.mean(c)
    else:
        c=np.mean(b-a)
    
    return c


### test functions
