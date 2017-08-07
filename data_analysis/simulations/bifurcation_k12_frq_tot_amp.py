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


def get_period(current_state):   
    """
    Take column of state variable from ode simulation
    calculate period by calculating distance between local maxima (see maxima_dist fct)
    Returns mean period of input variable
    """
    period = maxima_dist(current_state) / 10.0
   # assert math.isnan(period) == False
    
    return period

    
def get_amp(current_state):  
    """
    Take column of state variable from ode simulation
    Calculate amplitude by subtracting local maxima from local minima and divide by two
    Returns mean of amplitudes of input variable
    """
    amp = (get_maxima(current_state) - get_minima(current_state)) / 2
    #assert math.isnan(amp) == False
    
    return amp

def get_phase(current_state, frq_mrna_state):
    """
    Take column of state variable from ode simulation
    Calculate phase by subtracting indices of local maxima from frq_mrna reference maxima indices
    Define phase to be always positive
    Normalize to 1 (input variables own period)
    Returns mean of phases of input variable
    """
    maxima_idx = get_maxima_idx(current_state)
    first_maxima_idx = maxima_idx[:10]
    frq_mrna_maxima_idx = get_maxima_idx(frq_mrna_state)
    frq_mrna_maxima_idx = frq_mrna_maxima_idx[:10]
    
    if first_maxima_idx.any() :
        phase = first_maxima_idx - frq_mrna_maxima_idx   
        if np.sum(phase) < 0 :
            phase = frq_mrna_maxima_idx - first_maxima_idx
    
        relative_phase = np.mean(phase) / maxima_dist(current_state)
        phase = relative_phase
        assert relative_phase >= 0      
    else:
        phase = 0
        
    #assert math.isnan(phase) == False
    
    return phase

def get_maxima_idx(current_state):
    """
    Take column of state variable from ode simulation
    calculate indices of local maxima
    Return array of local maxima
    """
    maxima_idx = np.ravel(np.array(argrelextrema(current_state, np.greater)))
    return maxima_idx


def get_maxima(current_state):
    """
    Take column of state variable from ode simulation
    calculate local maxima
    Returns mean of local maxima
    """
    maxima = current_state[argrelextrema(current_state, np.greater)[0]]
    if maxima.size:
        return np.mean(maxima)
    else:
        return np.mean(current_state)

def get_minima(current_state):
    """
    Take column of state variable from ode simulation
    calculate local minima
    Returns mean of local minima
    """
    minima = current_state[argrelextrema(current_state, np.less)[0]]
    if minima.size:
        return np.mean(minima)
    else:
        return np.mean(current_state) 

def maxima_dist(current_state):
    """
    Take column of state variable from ode simulation
    calculate distance between maxima
    Return mean of distance between maxima
    """
    maxima_dist = np.mean(np.diff(get_maxima_idx(current_state)))
    if maxima_dist.any():
        return maxima_dist
    else: 
        return 0
    
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


##############################################################################
##############################################################################
##############################################################################
### bifurcation analysis

### initialize parameter array
bif_array = np.linspace(0,0.1,100)


#### dummy arrays to be filled after simulation steps
frq_tot_max_array = np.empty_like(bif_array)
frq_tot_min_array = np.empty_like(bif_array)

period_frq_tot_array = np.empty_like(bif_array)

params = rate.copy()
param = 'k12'

for idx, valx in enumerate(bif_array):
    params[param] = valx
    state = odeint(clock,state0,t,args=(params,))
    state_notrans = remove_trans(state)
    
    ### store maxima and minima after each simulation step
    frq_tot = state_notrans[:,1] + state_notrans[:,2]
    frq_tot_max_array[idx] = get_maxima(frq_tot)
    frq_tot_min_array[idx] = get_minima(frq_tot)
    
    ### criterion for the period to be defined
    if (frq_tot_max_array[idx] - frq_tot_min_array[idx] > 5):
        period_frq_tot_array[idx] = get_period(frq_tot)
    else: period_frq_tot_array[idx] = np.nan

##############################################################################
##############################################################################
### plot the bifurcation
datestring = datetime.strftime(datetime.now(), '%Y-%m-%d')
save_to = 'C:/Users/Philipp/Desktop/neurospora/figures/bifurcations/frq_tot/'


plt.figure(figsize=(20,10))
xlabel = param
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

plt.savefig(save_to + datestring + "-" + param)
plt.show()