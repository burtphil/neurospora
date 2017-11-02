# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:46:21 2017

@author: Philipp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.collections as collections

pi = np.pi

def ztan(t,T,kappa, s=10, z0 = 0.07):
    pi = np.pi
    om = 2*pi/T
    mu = pi/(om*np.sin(kappa*pi))
    cos1 = np.cos(om*t)
    cos2 = np.cos(kappa*pi)
    out = 1+2*z0*((1/pi)*np.arctan(s*mu*(cos1-cos2)))
    return out

def get_extrema(y,t):
    """
    take two arrays: y values and corresponding time array
    finds local maxima and minima
    finds adjacent values next to local maxima and minima
    return list with maxima and minima
    both list entries contain three arrays corresponding to actual extrema, and both neighbors
    """
    assert y.size > 3, "y array passed to get_extrema not large enough"
    
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

def get_max(a,t):
    ex = get_extrema(a,t)[0]
    out = ipol(ex,1)
    return out

def get_min(a,t):
    ex = get_extrema(a,t)[1]
    out = ipol(ex,1)
    return out

def get_xmax(a,t):
    ex = get_extrema(a,t)[0]
    out = ipol(ex,0)
    return out

def get_xmin(a,t):
    ex = get_extrema(a,t)[1]
    out = ipol(ex,0)
    return out

def ipol(arr,nr):
    l=[]
    for x in arr:
        l.append(interpolate(x)[nr])
    return np.asarray(l)

def get_periods(a,t):
    """
    take array containing time stamps of maxima
    returns the mean of time differences between maxima
    """
    ex = get_extrema(a,t)[1]
    
    l = ipol(ex,0)
    
    diff = np.diff(l)
    
    return diff


def get_phase(a,ta, b, tb):
    """
    takes two np arrays (containing maxima)
    choose first 10 maxima in both arrays
    subtract and check if entries are positive
    if not, subtract other way around
    return phase 2pi normalized over zeitgeber period
    """
    a = get_xmin(a,ta)
    b = get_xmin(b,tb)
    a = a[:10]
    b = b[:10]
    c = a-b
    if np.sum(c)>0:
        c=np.mean(c)
    else:
        c=np.mean(b-a)
    
    return c


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

def clock(state, t, rate, T, kappa):
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
        
        dt_frq_mrna     = (ztan(t,T,kappa) * rate['k1'] * (wc1_n**2) / (rate['K'] + (wc1_n**2))) - (rate['k4'] * frq_mrna) 
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

### simulation parameters

T = 22.0
kappa = 0.5
warm_dur = kappa*T

t = np.arange(0,120*T,0.1)

### make arrays containing only the last part of simulation
### run simulation
state = odeint(clock,state0,t,args=(rate,T,kappa))  

t_cut = t[int(-(50*T)):]
state_cut = state[int(-(50*T)):,1]
state_cut2 = state[int(-(50*T)):,0]

### lets make a thermoperiod array
thermoperiod = np.array([0.00001, 0.25, 0.5, 0.75, 1.0])
states = []
frq_mrna = []
frq_protein = []
colors = []

for i in thermoperiod:
    states.append(odeint(clock,state0,t,args=(rate,T,i)))
    colors.append(np.where((t_cut % T) <= (i*T),'tab:orange',np.where((t_cut % T) > (i*T),'k','k')))

for i in states:
    frq_mrna.append(i[int(-(50*T)):,0])
    frq_protein.append(i[int(-(50*T)):,1])

plot_name = ["Continuous cold", "25 % thermoperiod", "50 % thermoperiod",
             "75 % thermoperiod", "Continuous warmth"]  

  




"""
#### plot whole simulation
fig, ax = plt.subplots(figsize = (12,9))
ax.plot(t,state[:,1],"k")
ax.set_xlabel("t [h]")
ax.set_ylabel("[FRQ]c [a.u.]")
ax.set_xlim(t[0], t[-1])
ax.set_xticks(np.arange(0, int(t[-1]), 200))
collection = collections.BrokenBarHCollection.span_where(
    t, ymin=100, ymax=-100, where= ((t % T) <= warm_dur), facecolor='gray', alpha=0.2)
ax.add_collection(collection)
plt.show()

### plot last part of simulation
fig2, ax2 = plt.subplots(figsize = (12,9))
ax2.plot(t_cut,state_cut,"k")
ax2.set_xlabel("t [h]")
ax2.set_ylabel("[FRQ]c [a.u.]")
ax2.set_xlim(t_cut[0], t_cut[-1])
ax2.set_xticks(np.arange(int(t_cut[0]), int(t_cut[-1]), T/2))
#ax2.set_xticks(np.arange(0, 1200, 12.0))
collection = collections.BrokenBarHCollection.span_where(
    t_cut, ymin=100, ymax=-100, where= ((t_cut % T) <= warm_dur), facecolor='gray', alpha=0.5)
ax2.add_collection(collection)
plt.show()
"""

col = np.where((t_cut % T) <= warm_dur,'tab:orange',np.where((t_cut % T) > warm_dur,'k','k'))
### plot phase space
fig3,ax3 = plt.subplots(figsize = (12,9))
ax3.scatter(state_cut2, state_cut, c = col, s = 8)
ax3.set_xlabel("frq mRNA [a.u.]", fontsize= 'xx-large')
ax3.set_ylabel("FRQc [a.u.]",fontsize= 'xx-large')
ax3.tick_params(labelsize = 'x-large')


fig, axes = plt.subplots(1,5,figsize= (18,5))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.scatter(frq_mrna[i],frq_protein[i],s = 4, c = colors[i])
    ax.set_title(plot_name[i])
    ax.set_xlim([0,8])
    ax.set_ylim([13,37])
    ax.tick_params(labelsize = 'x-large')
fig.text(0.5, 0.01, 'frq mRNA', ha='center', fontsize = 'xx-large')
fig.text(0.08, 0.5, 'FRQc', va='center', rotation='vertical', fontsize = 'xx-large')
fig.savefig("phase_plane.pdf",bbox_inches='tight', dpi = 1200)



