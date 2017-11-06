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

def ztan(t,T,zstr, s=10, kappa = 0.5):
    pi = np.pi
    om = 2*pi/T
    mu = pi/(om*np.sin(kappa*pi))
    cos1 = np.cos(om*t)
    cos2 = np.cos(kappa*pi)
    out = 1+2*zstr*((1/pi)*np.arctan(s*mu*(cos1-cos2)))
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

def clock(state, t, rate, T, zstr):
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
        
        dt_frq_mrna     = (ztan(t,T,zstr) * rate['k1'] * (wc1_n**2) / (rate['K'] + (wc1_n**2))) - (rate['k4'] * frq_mrna) 
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
zstr = 0.1


t = np.arange(0,120*T,0.1)

### make arrays containing only the last part of simulation
### run simulation
state = odeint(clock,state0,t,args=(rate,T,zstr))  

t_cut = t[int(-(50*T)):]
state_cut = state[int(-(50*T)):,1]


### lets make a thermoperiod array

#### plot whole simulation
fig, ax = plt.subplots(figsize = (12,9))
ax.plot(t,state[:,1],"k")
ax.set_xlabel("t [h]", fontsize = 'xx-large')
ax.set_ylabel("FRQc [a.u.]", fontsize = 'xx-large')
ax.set_xlim(t[0], t[-1])
ax.set_xticks(np.arange(0, int(t[-1]), 200))
collection = collections.BrokenBarHCollection.span_where(
    t, ymin=100, ymax=-100, where= ((t % T) <= (T*0.5)), facecolor='gray', alpha=0.2)
ax.add_collection(collection)
ax.tick_params(labelsize = 'x-large')
plt.show()

### plot last part of simulation
fig2, ax2 = plt.subplots(figsize = (12,4))
ax2.plot(t_cut,state_cut,"k")
ax2.set_xlabel("t [h]", fontsize = 'xx-large')
ax2.set_ylabel("FRQc [a.u.]", fontsize = 'xx-large')
ax2.set_xlim(t_cut[0], t_cut[-1])
ax2.set_xticks(np.arange(int(t_cut[0]), int(t_cut[-1]), T/2))
#ax2.set_xticks(np.arange(0, 1200, 12.0))
collection = collections.BrokenBarHCollection.span_where(
    t_cut, ymin=100, ymax=-100, where= ((t_cut % T) <= (T*0.5)), facecolor='gray', alpha=0.5)
ax2.add_collection(collection)
ax2.tick_params(labelsize = 'x-large')
ax2.set_title("z0=0.1, T = 22 h")
plt.tight_layout()
fig2.savefig("trace_zstr.pdf", dpi = 1200)
plt.show()

### simulate different traces for frq1


def func(T,zstr):
    """
    Take zeitgeber cycle and thermoperiod as argument
    simulate system for 120 temp cycles
    cut off everything but last 5 temp cycles
    return cut off trace, corresponding time points and warm dur
    as list
    """
    t = np.arange(0,120*T,0.1)
    state = odeint(clock,state0,t,args=(rate,T,zstr))
    trace = state[int(-(50*T)):,1]
    t_cut = t[int(-(50*T)):]
    warm_dur = kappa*T
    
    return [t_cut,trace,warm_dur]
    
fig, axes = plt.subplots(4,1,figsize = (12,9))
axes = axes.flatten()
ax = axes[0]
ax1 = axes[1]
#ax2 = axes[2]
ax3 = axes[2]
ax4 = axes[3]

### show 1:1 entrainment for kappa = 0.5
T       = 22.0
kappa   = 0.5
trace   = func(T,kappa)[1]
t       = func(T,kappa)[0]
warm_dur = func(T,kappa)[2]

ax.plot(t,trace,"k")
ax.set_xlim(t[0], t[-1])
ax.set_xticks(np.arange(int(t[0]), int(t[-1]), T/2))
#ax2.set_xticks(np.arange(0, 1200, 12.0))
collection = collections.BrokenBarHCollection.span_where(
    t, ymin=100, ymax=-100, where= ((t % T) <= (T*0.5)), facecolor='gray', alpha=0.5)
ax.add_collection(collection)
ax.tick_params(labelsize = 'x-large')
ax.set_title("1:1 entrained, T=22, k = 0.5")
### show 1:1 entrainment for kappa = 0.75

T       = 22.0
kappa   = 0.25
trace   = func(T,kappa)[1]
t       = func(T,kappa)[0]
warm_dur = func(T,kappa)[2]

ax1.plot(t,trace,"k")
ax1.set_xlim(t[0], t[-1])
ax1.set_xticks(np.arange(int(t[0]), int(t[-1]), T/2))
#ax2.set_xticks(np.arange(0, 1200, 12.0))
collection = collections.BrokenBarHCollection.span_where(
    t, ymin=100, ymax=-100, where= ((t % T) <= (T*0.5)), facecolor='gray', alpha=0.5)
ax1.add_collection(collection)
ax1.tick_params(labelsize = 'x-large')
ax1.set_title("1:1 entrained, T=22, k = 0.25")
"""
### show 1:1 entrainment for T = 20 kappa = 0.5

T       = 20.0
kappa   = 0.5
trace   = func(T,kappa)[1]
t       = func(T,kappa)[0]
warm_dur = func(T,kappa)[2]

ax2.plot(t,trace,"k")
ax2.set_xlim(t[0], t[-1])
ax2.set_xticks(np.arange(int(t[0]), int(t[-1]), T/2))
#ax2.set_xticks(np.arange(0, 1200, 12.0))
collection = collections.BrokenBarHCollection.span_where(
    t, ymin=100, ymax=-100, where= ((t % T) <= warm_dur), facecolor='gray', alpha=0.5)
ax2.add_collection(collection)
ax2.tick_params(labelsize = 'x-large')
"""
### show 1:2 entrainment for T=11, kappa = 0.5
T       = 11.0
kappa   = 0.5
trace   = func(T,kappa)[1]
t       = func(T,kappa)[0]
warm_dur = func(T,kappa)[2]

ax3.plot(t,trace,"k")
ax3.set_xlim(t[0], t[-1])
ax3.set_xticks(np.arange(int(t[0]), int(t[-1]), T/2))
#ax2.set_xticks(np.arange(0, 1200, 12.0))
collection = collections.BrokenBarHCollection.span_where(
    t, ymin=100, ymax=-100, where= ((t % T) <= (T*0.5)), facecolor='gray', alpha=0.5)
ax3.add_collection(collection)
ax3.tick_params(labelsize = 'x-large')
ax3.set_title("1:2 entrained, T=11, k = 0.5")
### show no entrainment for T=26, kappa = 0.5
T       = 26.0
kappa   = 0.5
trace   = func(T,kappa)[1]
t       = func(T,kappa)[0]
warm_dur = func(T,kappa)[2]

ax4.plot(t,trace,"k")
ax4.set_xlim(t[0], t[-1])
ax4.set_xticks(np.arange(int(t[0]), int(t[-1]), T/2))
#ax2.set_xticks(np.arange(0, 1200, 12.0))
collection = collections.BrokenBarHCollection.span_where(
    t, ymin=100, ymax=-100, where= ((t % T) <= (T*0.5)), facecolor='gray', alpha=0.5)
ax4.add_collection(collection)
ax4.tick_params(labelsize = 'x-large')
ax4.set_title("Not entrained, T=26, k = 0.5")

fig.tight_layout()

fig.savefig("traces.pdf", dpi = 1200)
plt.show()    
    

