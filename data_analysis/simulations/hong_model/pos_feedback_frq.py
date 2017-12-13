# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/burt/.spyder2/.temp.py
"""
### import packages

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import string


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

t      = np.arange(0,120,0.1)

### what is a proper time resolution?

### run simulation
state = odeint(clock,state0,t,args=(rate,))  

### plot all ODEs
state_names = ["frq mRNA","FRQc","FRQn","wc-1 mRNA","WC-1c","WC-1n","FRQn:WC-1n"]

pos_frq = state[:,1] / (1+ state[:,1])

def style(axes):
    for ax in axes:
        ax.set_ylabel("$FRQ_c:(K+FRQ_c)$", fontsize = 'xx-large')        
        ax.tick_params(labelsize= 'x-large')

fig, axes = plt.subplots(1,2,figsize = (12,4))

axes.flatten()
ax = axes[0]
ax2 = axes[1]

ax.plot(t,pos_frq, 'k')
ax.set_ylim(0.9,1)
ax.set_yticks([0.9,0.95,1])
ax.set_xlabel("time (h)", fontsize = 'xx-large')
ax.set_xticks([0,60,120])

ax2.plot(state[:,1],pos_frq, 'k')
ax2.set_xlabel("$FRQ_c$", fontsize = "xx-large")
ax2.set_yticks([0,1,2])
ax2.set_ylim(0,2)
ax2.set_xticks([15,25,35])

style([ax,ax2])

ax.text(-0.17, .98, "A", transform=ax.transAxes, 
    size=20, weight='bold')

ax.text(1.03, .98, "B", transform=ax.transAxes, 
    size=20, weight='bold')


    
#ax1.text(1, .98, "B", transform=ax.transAxes, size=20, weight='bold')
#ax2.text(2.3, .98, "C", transform=ax.transAxes, size=20, weight='bold')

plt.tight_layout()
fig.savefig("pos_feedback_FRQ.pdf",dpi=1200)
plt.show()
