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
import sys
sys.path.append("C:\\Users\\Philipp\\Desktop\\neurospora\\data_analysis\\simulations")
import first_class as pb
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
bif_array = np.linspace(0,1,100)


#### dummy arrays to be filled after simulation steps
max_array = np.empty_like(bif_array)
min_array = np.empty_like(bif_array)

period_array = np.empty_like(bif_array)

params = rate.copy()

for idx, valx in enumerate(bif_array):
    params['k3'] = valx
    state = odeint(clock,state0,t,args=(params,))
    state_notrans = remove_trans(state)
    
    ### store maxima and minima after each simulation step
    frq_tot = state_notrans[:16000,1]
    max_array[idx] = np.mean(pb.get_max(frq_tot, tref))
    min_array[idx] = np.mean(pb.get_min(frq_tot, tref))
    
    ### criterion for the period to be defined
    if (max_array[idx] - min_array[idx] > 5):
        period_array[idx] = np.mean(pb.get_periods(frq_tot, tref))
    else: period_array[idx] = np.nan

##############################################################################
##############################################################################

### plot the bifurcation

xlabel = "rate of FRQ import to nucleus, k3"
datestring = datetime.strftime(datetime.now(), '%Y-%m-%d')
save_to = 'C:/Users/Philipp/Desktop/neurospora/figures/bifurcations/frq_tot/'

plt.figure(figsize=(12,8))

plt.subplot(121)
plt.plot(bif_array, max_array, 'k', bif_array, min_array, 'k')
plt.xlabel(xlabel)
plt.ylabel("$[FRQ]_{tot}$, a.u.")

plt.subplot(122)
plt.plot(bif_array, period_array)
plt.xlabel(xlabel)
plt.ylabel("period, h")
plt.ylim([14,28])
plt.tight_layout()

plt.savefig(save_to + datestring + "-" + "k3")
plt.show()