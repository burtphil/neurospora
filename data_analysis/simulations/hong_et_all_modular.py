#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:00:15 2017

@author: burt
"""

#### define all dictionaries used


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelextrema

### dictionary of parameters

### rate constants per hour
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
    'K2'    : 1.0
    
}


#### dummy dictionaries

period_dict = {}
amp_dict = {}
phase_dict = {}
params_plus = {}
params_minus = {}



#### functions
##############################################################################
##############################################################################
##############################################################################
def write_var(state):
    get_amp(state)
    get_period(state)
    get_phase(state)
    

def make_amp_dict(state):
    """
    Take state variable from ode simulation
    Calls amplitude function
    Returns dictionary of amplitudes
    """
    amp_dict = {}
    no_trans = remove_trans(state)
    for idx, valx in enumerate(state_names):
        current_state = no_trans[:,idx]
        amp_dict[valx] = get_amp(current_state)
    return amp_dict    
    
def make_phase_dict(state):
    """
    Take state variable from ode simulation
    Calls phase function
    Returns dictionary of phases
    """    
    phase_dict = {}
    no_trans = remove_trans(state)
    frq_mrna_state = no_trans[:,0]
    for idx, valx in enumerate(state_names):
        current_state = no_trans[:,idx]
        phase_dict[valx]= get_phase(current_state, frq_mrna_state)
    return phase_dict

def make_period_dict(state):
    """
    Take state variable from ode simulation
    Calls period function
    Returns dictionary of periods
    """
    period_dict = {}
    no_trans = remove_trans(state)  
    for idx, valx in enumerate(state_names):
        current_state = no_trans[:,idx]
        period_dict[valx]=get_period(current_state)
    return period_dict

def get_period(current_state):   
    """
    Take column of state variable from ode simulation
    calculate period by calculating distance between local maxima (see maxima_dist fct)
    Returns period of input variable
    """
    period = maxima_dist(current_state) / 10.0
    return period

    
def get_amp(current_state):  
    """
    Take column of state variable from ode simulation
    Calculate amplitude by subtracting local maxima from local minima and divide by two
    Returns mean of amplitudes of input variable
    """
    amp = (get_maxima(current_state) - get_minima(current_state)) / 2
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
        phase = np.nan
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
    return np.mean(maxima)

def get_minima(current_state):
    """
    Take column of state variable from ode simulation
    calculate local minima
    Returns mean of local minima
    """
    minima = current_state[argrelextrema(current_state, np.less)[0]]
    return np.mean(minima)

def maxima_dist(current_state):
    """
    Take column of state variable from ode simulation
    calculate distance between maxima
    Return mean of distance between maxima
    """
    maxima_dist = np.mean(np.diff(get_maxima_idx(current_state)))
    return maxima_dist
    
def remove_trans(state):
    """
    Take state variable from ode simulation
    Remove transients from state variable
    Return state variable without transients
    """
    return np.array(state[1600:,:])

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
        
        dt_frq_mrna     = (rate['k1'] * (wc1_n**2) / (rate['K'] + (wc1_n**2))) - (rate['k4'] * frq_mrna) 
        dt_frq_c        = rate['k2'] * frq_mrna - ((rate['k3'] + rate['k5']) * frq_c)
        dt_frq_n        = (rate['k3'] * frq_c) + (rate['k14'] * frq_n_wc1_n) - (frq_n * (rate['k6'] + (rate['k13'] * wc1_n)))
        dt_wc1_mrna     = rate['k7'] - (rate['k10'] * wc1_mrna)
        dt_wc1_c        = (rate['k8'] * frq_c * wc1_mrna / (rate['K2'] + frq_c)) - ((rate['k9'] + rate['k11']) * wc1_c)
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

t      = np.arange(0,480,0.1)

### what is a proper time resolution?

### run simulation
state = odeint(clock,state0,t,args=(rate,))  

state_names = ["frq mRNA","FRQc","FRQn","wc-1 mRNA","WC-1c","WC-1n","FRQn:WC-1n"]

### run a simulation varying each parameter of the model susequently by +- 10 percent

for key in rate:
    params_plus = rate.copy()
    params_minus = rate.copy()
    params_plus[key] = params_plus[key] * 1.1
    params_minus[key] = params_minus[key] * 0.9
    state_plus = odeint(clock,state0,t,args=(params_plus,))
    state_minus = odeint(clock,state0,t,args=(params_minus,))
    print make_phase_dict(state_plus)

    
 
##############################################################################
##############################################################################
##############################################################################
### plot the basic simulation
   
plt.plot(t,state)
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.legend(state_names,loc='center left', bbox_to_anchor=(0.6, 0.5))
plt.show()


###
plt.figure(figsize=(8,12))
plt.subplot(4,2,1)
plt.plot(t, state[:,0])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.title('frq mRNA')

plt.subplot(4,2,2)
plt.plot(t, state[:,1])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.title('FRQc')

plt.subplot(4,2,3)
plt.plot(t, state[:,2])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.title('FRQn')

plt.subplot(4,2,4)
plt.plot(t, state[:,3])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.title('wc-1 mRNA')

plt.subplot(4,2,5)
plt.plot(t, state[:,4])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.title('WC-1c')

plt.subplot(4,2,6)
plt.plot(t, state[:,5])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.title('WC-1n')

plt.subplot(4,2,7)
plt.plot(t, state[:,6])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.title('FRQn:WC-1n')
plt.tight_layout()
plt.show()