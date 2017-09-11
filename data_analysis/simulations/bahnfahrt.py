# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 15:58:45 2017

@author: Philipp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
### dictionary of parameters

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

def z(t):
    out = 1 + zeitgeber[idx]*np.cos(2*np.pi*t/tau[idy])
    return out

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
        
        dt_frq_mrna     = (z(t) * rate['k1'] * (wc1_n**2) / (rate['K'] + (wc1_n**2))) - (rate['k4'] * frq_mrna) + rate['k01'] 
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

### do one test simulation
t      = np.arange(0,3000,0.1)
state = odeint(clock,state0,t,args=(rate,))

############ study entrainment
state_names = ["frq mRNA",
               "FRQc",
               "FRQn",
               "wc-1 mRNA",
               "WC-1c",
               "WC-1n",
               "FRQn:WC-1n"]

### define variables for arnold tongue
zeitgeber = np.linspace(0,0.1,10)
tau = np.linspace(16,28,10)
zeit_mesh,warm_mesh = np.meshgrid(zeitgeber,tau)
entrain_mesh = np.zeros_like(zeit_mesh)

### simulate arnold tongue
for idx, valx in enumerate(zeitgeber):
    for idy, valy in enumerate(tau):
 
        ### define t so that there are enough temp cycles to cut out transients
        ### considering highest tau value
        t      = np.arange(0,3000,0.1)        
        state = odeint(clock,state0,t,args=(rate,))
              
        ### find time after 85 temp cycles (time res. is 0.1)
        ### then get system state x0 after 85 temp cycles
        t_state = int(85 * 10 * tau[idy])                        
        x0 = state[t_state,:]   
        
        ### remove transients
        n_state = state[t_state:,:]   

 
        ### define final entrainment criterion
        entrain_crit = times_mean / tau
        entrain = 0.5
        if entrain_crit < 1.1 and entrain_crit > 0.9:
            entrain = 1
        else:
            entrain = 0
            
        print("zeitgeber period = ",tau)
        print("thermoperiod = ", valy)
        print("entrain = ", entrain)
#        print("entrain_crit = ",entrain_crit)
#        print("entrain = ",entrain)
#        print("t_state = ",t_state)
#        print("run_time = ", run_time)
        print("")
        
        entrain_mesh[idy,idx] = entrain

entr = entrain_mesh[:-1,:-1]
plt.pcolormesh(zeit_mesh,warm_mesh, entr, edgecolors = "k", vmin = 0, vmax = 1)
plt.colorbar()
plt.show()