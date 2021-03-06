# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:46:21 2017

@author: Philipp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import colorcet as cc
from datetime import datetime

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

#rate constants for original model
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

### rate constants for frq1
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


def tongue(zeitgeber, T, upper = 1, lower = 1, res = 4000, phase = 0, tcycle = 85):
    
    ratio = upper/lower
    zeit_mesh,warm_mesh = np.meshgrid(zeitgeber,T)
    entrain_mesh = np.zeros_like(zeit_mesh)

    ### simulate arnold tongue
    for idx, valx in enumerate(zeitgeber):
        for idy, valy in enumerate(T):
            
            ### set local variables for z(t) function
            global zstr 
            global iper
            zstr= zeitgeber[idx]
            iper = T[idy]

            ### define t so that there are enough temp cycles to cut out transients
            ### considering highest tau value
            t       = np.arange(0,res,0.1)        
            state   = odeint(clock,state0,t,args=(rate,))
                  
            ### find time after 85 temp cycles (time res. is 0.1)
            ### then get system state x0 after 85 temp cycles
            t_state = int(tcycle * 10 * T[idy])                        
            x0      = state[t_state:,1]
            tn      = t[t_state:]
            z_state = z(t)
            z0      = z_state[t_state:]
            ### do the same for extrinsic zeitgeber function

            #z0      = z_state[t_state:]
            
            ### get extrema and neighbors for zeitgeber function and simulated data
            frq_per  = get_periods(x0, tn)       
            period = np.mean(frq_per)        
            entr = 2*np.pi
            
            ### define entrainment criteria
            ### T-tau should be < 5 minutes
            ### period std should be small
            c = np.abs(T[idy]-ratio*period)*60
            c2 = np.std(np.diff(frq_per))
            
            if c < 5 and c2 < 5:
                if phase == 0:
                    print "entrained!"
                    entr = 0
                else:
                    if zstr != 0:
                        ph = get_phase(x0,tn,z0,tn)
                ### normalize phase to 2pi and set entr to that phase
                        entr = 2*np.pi*ph/iper
                
                    else: entr = 0
            print idx
            print idy
            print ""
            
            entrain_mesh[idy,idx] = entr
    
    
    ent = entrain_mesh[:-1,:-1]
    
    
    date = datetime.strftime(datetime.now(), '%Y_%m_%d')
    
    t_name = '_tongue_'+str(int(upper))+'_'+str(int(lower))+"_z_"+str(round(zeitgeber[-1],1))
    
    if phase == 1: 
        t_name = t_name + "_ph"
        
    save_to = '/home/burt/neurospora/figures/entrainment/'
    
    
    np.savez(save_to+date+t_name, warm_mesh = warm_mesh, zeit_mesh = zeit_mesh, ent = ent)

    fig, ax = plt.subplots(figsize=(12,9))
    heatmap = ax.pcolormesh(warm_mesh,zeit_mesh, ent, cmap = cc.m_fire, edgecolors = "none", vmin = 0, vmax = 2*np.pi)
 
    ax.set_xlabel("T [h]", fontsize =22)
    ax.set_ylabel("Z [a.u.]", fontsize =22)
    ax.tick_params(labelsize = 16)
    
    if phase == 1:
        cbar = fig.colorbar(heatmap,ticks=[0,np.pi/2,np.pi,1.5*np.pi,2*np.pi], label = 'Phase [rad]')
        cbar.ax.set_yticklabels(['0','$\pi/2$','$\pi$','$3\pi/2$', '2$\pi$'])
        cbar.ax.tick_params(labelsize = 16)
        cbar.ax.set_ylabel("Phase [rad]", fontsize = 22)
        

    plt.tight_layout()
    fig.savefig(save_to+date+t_name+".pdf", dpi=1200)
    plt.show()




#### 1/1 tongue
### plot 1/1 with long transients in black and with phase



tcycle = 100

#### 1/1 tongue
"""
mi = 24.0
ma = 34.0
upper = 1.0
lower = 1
res = 7000
T = np.linspace(mi,ma,100)

zeitgeber = np.linspace(0.0,0.1,100)
tongue(zeitgeber = zeitgeber, T = T,upper = upper,lower = lower,res = res, phase = 1, tcycle = tcycle)
"""
"""
#### 1/2 tongue
mi = 7.0
ma = 15.0
upper = 1.0
lower = 2.0
res = 5000
T = np.linspace(mi,ma,100)

tongue(zeitgeber = zeitgeber, T = T,upper = upper,lower = lower,res = res, phase = 0, tcycle = tcycle)
"""
#### 2/1 tongue

mi = 39.0
ma = 41.0
upper = 2.0
lower = 1.0
res = 7000
zeitgeber = np.linspace(0.34,0.42,10)
T = np.linspace(mi,ma,10)

tongue(zeitgeber = zeitgeber, T = T,upper = upper,lower = lower,res = res, phase = 0, tcycle = tcycle)


"""
### 3 : 2 tongue
mi = 30.0
ma = 36.0
upper = 3.0
lower = 2.0
res = 7000
T = np.linspace(mi,ma,50)

zeitgeber = np.linspace(0,1,50)
tongue(zeitgeber = zeitgeber, T = T,upper = upper,lower = lower,res = res, phase = 0, tcycle = tcycle)

### 2 : 3 tongue
mi = 13.5
ma = 16.0
upper = 2.0
lower = 3.0
res = 5000
T = np.linspace(mi,ma,50)

zeitgeber = np.linspace(0,1,50)
tongue(zeitgeber = zeitgeber, T = T,upper = upper,lower = lower,res = res, phase = 0, tcycle = tcycle)

### 1 : 3 tongue
mi = 6.5
ma = 10.0
upper = 1.0
lower = 3.0
res = 5000
T = np.linspace(mi,ma,50)

zeitgeber = np.linspace(0,1,50)
tongue(zeitgeber = zeitgeber, T = T,upper = upper,lower = lower,res = res, phase = 0, tcycle = tcycle)

### 3 : 1 tongue
mi = 60.0
ma = 70.0
upper = 3.0
lower = 1.0
res = 9000
T = np.linspace(mi,ma,50)

zeitgeber = np.linspace(0,1,50)
tongue(zeitgeber = zeitgeber, T = T,upper = upper,lower = lower,res = res, phase = 0, tcycle = tcycle)
"""

