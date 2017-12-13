#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:14:09 2017

@author: burt
"""
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

rate1 = {
    'k1'    : 1.8,
    'k2'    : 1.8,
    'k3'    : 0.15,
    'k4'    : 0.23,
    'k5'    : 0.4,
    'k6'    : 0.1,
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

### rate constants for frq7
rate7 = {
    'k1'    : 1.8,
    'k2'    : 1.8,
    'k3'    : 0.05,
    'k4'    : 0.23,
    'k5'    : 0.15,
    'k6'    : 0.01,
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

t      = np.arange(0,240,0.1)

### what is a proper time resolution?

### run simulation
state = odeint(clock,state0,t,args=(rate,))  
state1 = odeint(clock,state0,t,args=(rate1,)) 
state7 = odeint(clock,state0,t,args=(rate7,)) 
### plot all ODEs
state_names = ["frq mRNA","$FRQ_c$","$FRQ_n$","wc-1 mRNA","$WC$-$1_c$",
               "$WC$-$1_n$","$FRQ_n$:$WC$-$1_n$"]


"""
###
plt.figure(figsize=(12,9))
plt.subplot(111)
plt.plot(t, state[:,1], label= "frq+")
plt.plot(t, state1[:,1], label = "frq1")
plt.plot(t, state7[:,1], label = "frq7")
plt.xlim((140,240))
plt.xlabel("time [h]", fontsize= 'xx-large')
plt.ylabel('FRQc', fontsize= 'xx-large')
plt.tick_params(labelsize= 'x-large')
plt.legend(loc = 1)

plt.savefig("simulation_strains.pdf",dpi=1200)
plt.tight_layout()
plt.show()

"""
fig, axes = plt.subplots(3,2, figsize = (12,9))
axes = axes.flatten()

ax1 = axes[0]
ax2 = axes[1]
ax3 = axes[2]
ax4 = axes[3]
ax5 = axes[4]
ax6 = axes[5]

ax1.plot(t,state)
#ax1.set_xlabel("time (h)", fontsize= 'xx-large')
ax1.set_ylabel("conc. (a.u.)", fontsize= 'xx-large')
#plt.xticks(np.arange(0, 49, 12.0))
ax1.legend(state_names,loc='upper right')

ax2.plot(t, state[:,0],"tab:blue", t, state[:,5],"tab:brown")
#plt.xlabel("time [h]", fontsize= 'xx-large')
ax2.set_ylabel('conc (a.u.)', fontsize= 'xx-large')
ax2.legend([state_names[0],state_names[5]], loc = 1)
#ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax2.set_yticks([0.0312,0.0316,0.0320,0.0324])
#plt.xticks(np.arange(0, 49, 12.0))
#plt.title('WC-1c')
ax3.plot(t, state[:,2],"g")
#plt.xlabel("time [h]", fontsize= 'xx-large')
ax3.set_ylabel('$FRQ_n$', fontsize= 'xx-large')
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax3.set_yticks([0.025,0.075,0.125])
#plt.xticks(np.arange(0, 49, 12.0))
#plt.title('FRQn')

ax4.plot(t, state[:,6],"tab:pink")
#ax4.set_xlabel("time (h)", fontsize= 'xx-large')
ax4.set_ylabel('$FRQ_n:WC$-$1_n$', fontsize= 'xx-large')
ax4.tick_params(labelsize= 'x-large')
ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax4.set_yticks([0.1,0.15,0.2])


ax5.plot(t, state[:,1],"k", label= "frq+")
ax5.plot(t, state1[:,1],"tab:gray", label = "frq1")
ax5.plot(t, state7[:,1],"k--", label = "frq7")
ax5.set_xlabel("time (h)", fontsize= 'xx-large')
ax5.set_ylabel('$FRQ_c$', fontsize= 'xx-large')
ax5.legend(loc = 1)

ax6.plot(t, state[:,1],"orange", label = '$FRQ_c$')
#plt.xlabel("time [h]", fontsize= 'xx-large')
ax6.set_ylabel('$FRQ_c$', fontsize= 'xx-large')
#ax6.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax6.set_yticks([0.025,0.075,0.125])
ax6.set_xlabel("time (h)", fontsize= 'xx-large')
ax7 = ax6.twinx()
ax7.plot(t, state[:,4],"purple", label = '$WC$-$1_c$')
ax7.set_ylabel('$WC$-$1_c$', fontsize= 'xx-large')
ax7.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax7.set_yticks([0.0312,0.0316,0.0320,0.0324])
ax7.tick_params(labelsize= 'x-large')
lines, labels = ax6.get_legend_handles_labels()
lines2, labels2 = ax7.get_legend_handles_labels()
ax7.legend(lines + lines2, labels + labels2, loc=1)


for n,ax in enumerate(axes):
    ax.tick_params(labelsize= 'x-large')
    ax.text(-0.2, .97, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=20, weight='bold')
    ax.set_xlim((140,240))
    
plt.tight_layout()

fig.savefig("first_fig.pdf",dpi =1200)
