# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:46:21 2017

@author: Philipp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

### global variables
### dummy variables for z(t) fct
zstr = 0
iper = 22

### define functions
def z(t):
    out = 1 + zstr*np.cos(2*np.pi*t/iper)
    return out

def get_extrema(y,t):
    """
    take two arrays: y values and corresponding time array
    finds local maxima and minima
    finds adjacent values next to local maxima and minima
    return list with maxima and minima
    both list entries contain three arrays corresponding to actual extrema, and both neighbors
    """

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

def per(arr):
    """
    take array containing time stamps of maxima
    returns the mean of time differences between maxima
    """
    diff = np.diff(arr)
    per = np.mean(diff)
    
    ### check that distribution of periods is not too wild
#    if np.std(diff) > 0.5 :
#        print "std is higher than 1"
    
    return per


def get_phase(a, b):
    """
    takes two np arrays (containing maxima)
    choose first 10 maxima in both arrays
    subtract and check if entries are positive
    if not, subtract other way around
    return phase 2pi normalized over zeitgeber period
    """
    a = a[:10]
    b = b[:10]
    c = a-b
    if np.sum(c)>0:
        c=np.mean(c)
    else:
        c=np.mean(b-a)
    ph = 2*np.pi*c/iper
    
    return ph
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

zeitgeber = np.linspace(0,0.1,100)
tau = np.linspace(16,28,100)
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
        ex  = get_extrema(x0, tn)
        z_ex = get_extrema(z0, tn)
        
        ### choose only maxima (get extrema returns minima and maxima)
        max  = ex[1]
        z_max = z_ex[1]
        max_ipol = []
        z_max_ipol = []
        
        ### take every element in max and maxz (a triple ofmaximum and neigbors)
        ### and interpolate. put result in an empty list
        for x in max:
            max_ipol.append(interpolate(x)[0])

        for x in z_max:
            z_max_ipol.append(interpolate(x)[0])
        
        max_ipol = np.asarray(max_ipol, dtype = np.float64)
        z_max_ipol = np.asarray(z_max_ipol, dtype = np.float64)
        
        period = per(max_ipol)
        
        entr = 2*np.pi
        
        ### define entrainment criteria
        ### T-tau should be < 5 minutes
        ### period std should be small
        c1 = np.abs(tau[idy]-period)*60
        c2 = np.std(np.diff(max_ipol))
        
        if c1 < 5 and c2 < 0.5 :
            if zstr != 0:
                entr = get_phase(max_ipol,z_max_ipol)
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

### wha is a proper time resolution?
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
