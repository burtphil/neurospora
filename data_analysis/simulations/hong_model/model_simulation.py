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

t      = np.arange(0,500,0.1)

### what is a proper time resolution?

### run simulation
state = odeint(clock,state0,t,args=(rate,))  

### plot all ODEs
state_names = ["frq mRNA","$FRQ_c$","$FRQ_n$","wc-1 mRNA","$WC$-$1_c$",
               "$WC$-$1_n$","$FRQ_n$:$WC$-$1_n$"]


ylabel = "FRQc [a.u.]"
plt.figure(figsize=(12,9))
ax = plt.subplot(111)
ax.plot(t,state)
ax.set_xlabel("time (h)", fontsize= 'xx-large')
ax.set_ylabel("concentrations (a.u.)", fontsize= 'xx-large')
#plt.xticks(np.arange(0, 49, 12.0))
plt.legend(state_names,loc='upper right')

ax.tick_params(labelsize= 'x-large')
ax.text(-0.05, .97, "A", transform=ax.transAxes, 
        size=20, weight='bold')
plt.tight_layout()
plt.show()
plt.savefig("simulation1.pdf",dpi=1200)

###
plt.figure(figsize=(9,12))
ax = plt.subplot(3,1,1)
ax.plot(t, state[:,4],"purple")
#plt.xlabel("time [h]", fontsize= 'xx-large')
ax.set_ylabel('$WC$-$1_c$', fontsize= 'xx-large')
ax.tick_params(labelsize= 'x-large')
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set_yticks([0.0312,0.0316,0.0320,0.0324])
#plt.xticks(np.arange(0, 49, 12.0))
#plt.title('WC-1c')

ax1 = plt.subplot(3,1,2)
ax1.plot(t, state[:,2],"g")
#plt.xlabel("time [h]", fontsize= 'xx-large')
ax1.set_ylabel('$FRQ_n$', fontsize= 'xx-large')
ax1.tick_params(labelsize= 'x-large')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.set_yticks([0.025,0.075,0.125])
#plt.xticks(np.arange(0, 49, 12.0))
#plt.title('FRQn')

ax2 = plt.subplot(3,1,3)
ax2.plot(t, state[:,6],"pink")
ax2.set_xlabel("time (h)", fontsize= 'xx-large')
ax2.set_ylabel('$FRQ_n:WC$-$1_n$', fontsize= 'xx-large')
ax2.tick_params(labelsize= 'x-large')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.set_yticks([0.1,0.15,0.2])

axes = [ax,ax1,ax2]

for n, ax in enumerate(axes): 
    ax.text(-0.11, .97, string.ascii_uppercase[n+1], transform=ax.transAxes, 
            size=20, weight='bold')
#plt.xticks(np.arange(0, 49, 12.0))
#plt.title('FRQn:WC-1n')
#plt.savefig("simulation2.pdf",dpi=1200)
plt.tight_layout()
plt.savefig("simulation2.pdf",dpi=1200)
plt.show()

"""
def style(axes):
    for ax in axes:
        ax.tick_params(labelsize="x-large")
        ax.set_ylabel("$FRQ_c$", fontsize = "xx-large")

fig = plt.figure(figsize=(12,9))

ax = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
ax2 = plt.subplot2grid((3, 2), (0, 1))
ax3 = plt.subplot2grid((3, 2), (1, 1))
ax4 = plt.subplot2grid((3, 2), (2, 1))

ax.plot(t,state)
ax.legend(state_names,loc='upper right')
ax2.plot(t, state[:,4],"purple")
ax3.plot(t, state[:,2],"g")
ax4.plot(t, state[:,6],"pink")

axes = [ax2,ax3,ax4]

style(axes)

for n, ax in enumerate(axes): 
    ax.text(-0.13, .97, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=20, weight='bold')
"""