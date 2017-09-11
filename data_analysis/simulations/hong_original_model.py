# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/burt/.spyder2/.temp.py
"""
### import packages

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelextrema

####### implement biological model Hong et al 2008

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

### define ODE clock function

def clock(state, t, rate):
        ### purpose:simulate Hong et al 2008 model for neuropora clock


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
               
### set initial state and time vector

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

### plot all ODEs
state_names = ["frq mRNA","FRQc","FRQn","wc-1 mRNA","WC-1c","WC-1n","FRQn:WC-1n"]

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


### make dummy to iterate over state
dummy = np.array([0,1,2,3,4,5,6])

### make empty dictionaries to later populate with amplitudes, periods and phases
amplitudes_dict = {}
periods_dict = {}
phases_dict = {}
mean_oscillation_dict = {}

### define a new state array to exlude transients in model
### exculde first 1600 ode iterations

state_notrans = np.array(state[1600:,:])

### calculate positions of frq mrna maxima to determine relative phases
frq_mrna_state = state_notrans[:,0]

### need to convert to 1d array using flatten
frq_mrna_maxima_idx = np.ravel(np.array(argrelextrema(frq_mrna_state, np.greater)))

### choose only first 10 maxima that occur
frq_mrna_maxima_idx = frq_mrna_maxima_idx[:10]

for idx, valx in enumerate(state_names):
    ### iterate over all columns in state vector and calculate mean and amplitude
    current_state = state_notrans[:,idx]
    ### take local maxima and minima and calculate mean
    ### define ampliude as max - min / 2
    local_maxima = current_state[argrelextrema(current_state, np.greater)[0]]
    local_minima = current_state[argrelextrema(current_state, np.less)[0]]
    local_maxima_mean = np.mean(local_maxima)
    local_minima_mean = np.mean(local_minima)   
    local_maxima_idx = np.ravel(np.array(argrelextrema(current_state, np.greater)))
    local_maxima_distance = np.mean(np.diff(local_maxima_idx))
    first_local_maxima_idx = local_maxima_idx[:10]
    
    ### get the position of the local maxima and calculate distance between them
    ### then take mean and divide by integration frequency which is set to 10
    periods_dict[valx] = local_maxima_distance / 10.0
    amplitudes_dict[valx] = (local_maxima_mean - local_minima_mean) / 2
    
    
    ### calculate the phase as the difference between two maxima
    ### choose first ten local maxima and calculate the difference to frq_mrna_maxima
    if first_local_maxima_idx.any() :
        phase = first_local_maxima_idx - frq_mrna_maxima_idx
    ### if the elements of the returned array are negative do it the other way around
    
        if np.sum(phase) < 0 :
            phase = frq_mrna_maxima_idx - first_local_maxima_idx

        relative_phase = np.mean(phase) / local_maxima_distance
        phases_dict[valx] = relative_phase
        assert relative_phase >= 0
    else:
        phases_dict[valx] = np.nan




