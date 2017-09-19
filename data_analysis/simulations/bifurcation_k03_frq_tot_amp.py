#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:00:15 2017

@author: burt
"""

#### define all dictionaries used


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelextrema
from datetime import datetime

### define model variable names used in dictionaries

state_names = ["frq mRNA","FRQc","FRQn","wc-1 mRNA","WC-1c","WC-1n",
               "FRQn:WC-1n"]


### dictionary of parameters

### rate constants per hour

### these parameters differ from the publication but were sent by email from peter ruoff
rate = {
    'k1'    : 1.8,
    'k2'    : 1.8,
    'k3'    : 0.05,
    'k4'    : 0.23,
    'k5'    : 0.27,
    'k6'    : 0.07,
    'k7'    : 0.5,
    'k8'    : 0.8,
    'k9'    : 40.0,
    'k10'   : 0.3,
    'k11'   : 0.05,
    'k12'   : 0.02,
    'k13'   : 50.0,
    'k14'   : 1.0,
    'k15'   : 8.0,
    'K'     : 1.25,
    'K2'    : 1.0,
    'k01'   : 0,
    'k02'   : 0
}

### old parameters
rate_old = {
    'k1'    : 1.8,
    'k2'    : 1.8,
    'k3'    : 0.05,
    'k4'    : 0.23,
    'k5'    : 0.27,
    'k6'    : 0.07,
    'k7'    : 0.16,
    'k8'    : 0.8,
    'k9'    : 40.0,
    'k10'   : 0.1,
    'k11'   : 0.05,
    'k12'   : 0.02,
    'k13'   : 50.0,
    'k14'   : 1.0,
    'k15'   : 5.0,
    'K'     : 1.25,
    'K2'    : 1.0,
    'k01'   : 0,
    'k02'   : 0
}

#### functions
##############################################################################
##############################################################################
##############################################################################
def get_extrema(y,t):
    """
    take two arrays: y values and corresponding time array
    finds local maxima and minima
    finds adjacent values next to local maxima and minima
    return list with maxima and minima
    both list entries contain three arrays corresponding to actual extrema, and both neighbors
    """

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

def get_max(arr):
    
    l= []
    for x in arr:
        l.append(interpolate(x)[0])
    return np.asarray(l)

def get_min(arr):
    
    l= []
    for x in arr:
        l.append(interpolate(x)[1])
    return np.asarray(l)

def get_period(arr):
    """
    take array containing time stamps of maxima
    returns the mean of time differences between maxima
    """
    diff = np.diff(arr)
    per = np.mean(diff)
    
    ### check that distribution of periods is not too wild
#    if np.std(diff) > 0.5 :
#        print "std is higher than 1"
    
    return per

    
def remove_trans(state):
    """
    Take state variable from ode simulation
    Remove transients from state variable
    Return state variable without transients
    """
    return np.array(state[16000:,:])

def clock(state, t, rate):

        ### purpose:simulate Hong et al 2008 model for neuropora clock
        """
        Take state variable from ode simulation
        Calls amplitude function
        Returns dictionary of amplitudes
        """

        ### define state vector

        frq_mrna    = state[0]
        frq_c       = state[1]
        frq_n       = state[2]
        wc1_mrna    = state[3]
        wc1_c       = state[4]
        wc1_n       = state[5]
        frq_n_wc1_n = state[6]
        
        
        
        ###  ODEs Hong et al 2008
        ### letzter summand unklar bei dtfrqmrna
        
        dt_frq_mrna     = (rate['k1'] * (wc1_n**2) / (rate['K'] + (wc1_n**2))) - (rate['k4'] * frq_mrna) + rate['k01'] 
        dt_frq_c        = rate['k2'] * frq_mrna - ((rate['k3'] + rate['k5']) * frq_c)
        dt_frq_n        = (rate['k3'] * frq_c) + (rate['k14'] * frq_n_wc1_n) - (frq_n * (rate['k6'] + (rate['k13'] * wc1_n)))
        dt_wc1_mrna     = rate['k7'] - (rate['k10'] * wc1_mrna)
        dt_wc1_c        = (rate['k8'] * frq_c * wc1_mrna / (rate['K2'] + frq_c)) - ((rate['k9'] + rate['k11']) * wc1_c) + (rate['k02'] * wc1_mrna)
        dt_wc1_n        = (rate['k9'] * wc1_c) - (wc1_n * (rate['k12'] + (rate['k13'] * frq_n))) + (rate['k14'] * frq_n_wc1_n)
        dt_frq_n_wc1_n  = rate['k13'] * frq_n * wc1_n - ((rate['k14'] + rate['k15']) * frq_n_wc1_n)
        
        ### derivatives
        
        dt_state = [dt_frq_mrna,
                    dt_frq_c,
                    dt_frq_n,
                    dt_wc1_mrna,
                    dt_wc1_c,
                    dt_wc1_n,
                    dt_frq_n_wc1_n]
        
        return dt_state
 
##############################################################################
##############################################################################
##############################################################################

### set initial conditions for each ODE

frq_mrna0    = 4.0
frq_c0       = 30.0
frq_n0       = 0.1
wc1_mrna0    = (0.5 / 0.3)
wc1_c0       = 0.03225
wc1_n0       = 0.35
frq_n_wc1_n0 = 0.18

state0 = [frq_mrna0,
          frq_c0,
          frq_n0,
          wc1_mrna0,
          wc1_c0,
          wc1_n0,
          frq_n_wc1_n0]

### set time to integrate

t      = np.arange(0,4800,0.1)
tref = t[:16000]



##############################################################################
##############################################################################
##############################################################################
### bifurcation analysis

### initialize parameter array
bif_array = np.linspace(0.03,0.27,100)


#### dummy arrays to be filled after simulation steps
frq_tot_max_array = np.empty_like(bif_array)
frq_tot_min_array = np.empty_like(bif_array)

period_frq_tot_array = np.empty_like(bif_array)

params = rate.copy()

for idx, valx in enumerate(bif_array):
    params['k3'] = valx
    state = odeint(clock,state0,t,args=(params,))
    state_notrans = remove_trans(state)
    
    ### store maxima and minima after each simulation step
    frq_tot = state_notrans[:,1]
    ex = get_extrema(frq_tot,tref)
    
    ymax = ex[1]
    ymin = ex[0]
    ymax_ipol = []
    ymin_ipol = []
    xmax_ipol = 
    for x in ymax:
        ymax_ipol.append(interpolate(x)[1])
    
    for y in ymin:
        ymin_ipol.append(interpolate(x)[1])
  
    ymax_ipol = np.asarray(ymax_ipol, dtype = np.float64)
    ymin_ipol = np.asarray(ymin_ipol, dtype = np.float64)
    
    frq_tot_max_array[idx] = np.mean(ymax_ipol)
    frq_tot_min_array[idx] = np.mean(ymin_ipol)
    
    ### criterion for the period to be defined
    if (frq_tot_max_array[idx] - frq_tot_min_array[idx] > 5):
        period_frq_tot_array[idx] = get_period(frq_tot)
    else: period_frq_tot_array[idx] = np.nan

##############################################################################
##############################################################################

### plot the bifurcation

xlabel = "rate of FRQ import to nucleus, k3"
datestring = datetime.strftime(datetime.now(), '%Y-%m-%d')
save_to = 'C:/Users/Philipp/Desktop/neurospora/figures/bifurcations/frq_tot/'

plt.figure(figsize=(20,10))

plt.subplot(121)
plt.plot(bif_array, frq_tot_max_array, 'k', bif_array, frq_tot_min_array, 'k')
plt.xlabel(xlabel)
plt.ylabel("$[FRQ]_{tot}$, a.u.")

plt.subplot(122)
plt.plot(bif_array, period_frq_tot_array)
plt.xlabel(xlabel)
plt.ylabel("period, h")
plt.ylim([14,28])
plt.tight_layout()

plt.savefig(save_to + datestring + "-" + "k3")
plt.show()