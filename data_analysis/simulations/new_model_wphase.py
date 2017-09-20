# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:46:21 2017

@author: Philipp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
sys.path.append("C:\\Users\\Philipp\\Desktop\\neurospora\\data_analysis\\simulations")
import first_class as pb
### global variables
### dummy variables for z(t) fct
zstr = 0
iper = 22

### define functions
def z(t):
    out = 1 + zstr*np.cos(2*np.pi*t/iper)
    return out

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
        
        dt_frq_mrna     = (z(t) * rate['k1'] * (wc1_n**2) / (rate['K'] + (wc1_n**2))) - (rate['k4'] * frq_mrna) 
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
"""
t      = np.arange(0,480,0.1)

### what is a proper time resolution?

### run simulation
state = odeint(clock,state0,t,args=(rate,))  
"""
### plot all ODEs
state_names = ["frq mRNA","FRQc","FRQn","wc-1 mRNA","WC-1c","WC-1n","FRQn:WC-1n"]
"""
plt.plot(t,state)
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.legend(state_names,loc='center left', bbox_to_anchor=(0.6, 0.5))
plt.show()
"""

zeitgeber = np.linspace(0,0.1,10)
tau = np.linspace(16,28,10)
zeit_mesh,warm_mesh = np.meshgrid(zeitgeber,tau)
entrain_mesh = np.zeros_like(zeit_mesh)

### simulate arnold tongue
for idx, valx in enumerate(zeitgeber):
    for idy, valy in enumerate(tau):
        
        ### set local variables for z(t) function
        zstr = zeitgeber[idx]
        iper = tau[idy]
        ### define t so that there are enough temp cycles to cut out transients
        ### considering highest tau value
        t       = np.arange(0,3000,0.1)        
        state   = odeint(clock,state0,t,args=(rate,))
              
        ### find time after 85 temp cycles (time res. is 0.1)
        ### then get system state x0 after 85 temp cycles
        t_state = int(85 * 10 * tau[idy])                        
        x0      = state[t_state:,1]
        tn      = t[t_state:]
        ### do the same for extrinsic zeitgeber function
        
        z_state = z(t)
        z0      = z_state[t_state:]
        
        ### get extrema and neighbors for zeitgeber function and simulated data
        frq_per  = pb.get_periods(x0, tn)       
        period = np.mean(frq_per)        
        entr = 2*np.pi
        
        ### define entrainment criteria
        ### T-tau should be < 5 minutes
        ### period std should be small
        c1 = np.abs(tau[idy]-period)*60
        c2 = np.std(np.diff(frq_per))
        
        if c1 < 5 and c2 < 0.5 :
            if zstr != 0:
                ph = pb.get_phase(x0,tn,z0,tn)
                ### normalize phase to 2pi and set entr to that phase
                entr = 2*np.pi*ph/iper
                
            else: entr = 0
                       
        print idx*idy
        
        entrain_mesh[idy,idx] = entr


ent = entrain_mesh[:-1,:-1]

fig, ax = plt.subplots()
heatmap = ax.pcolormesh(warm_mesh,zeit_mesh, ent, cmap = "hot", edgecolors = "none", vmin = 0, vmax = 2*np.pi)
cbar = fig.colorbar(heatmap,ticks=[0,np.pi/2,np.pi,1.5*np.pi,2*np.pi], label = 'Phase [rad]')
cbar.ax.set_yticklabels(['0','$\pi/2$','$\pi$','$3\pi/2$', '2$\pi$'])
plt.xlabel("T [h]")
plt.ylabel("Z [a.u.]")
plt.show()
        
"""
t      = np.arange(0,3000,0.1)

### wha is a proper time resolution?i
zstr = 0.1
iper = 23
### run simulation
state = odeint(clock,state0,t,args=(rate,)) 
t_state = int(85 * 10 * iper) 
x0      = state[t_state:,1]
tn      = t[t_state:]


z_state = z(t)
z0 = z_state[t_state:]*115-90

plt.plot(tn,x0, tn, z0, "r")
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.show()
"""
