# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:46:21 2017

@author: Philipp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#import colorcet as cc
#from datetime import datetime
#import matplotlib.collections as collections

from scipy.signal import argrelextrema, hilbert, chirp
from scipy import interpolate

def ztan(t,T,z0,kappa, s=1):
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

def clock(state, t, rate, T, z0, kappa,signal=ztan):
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
        
        dt_frq_mrna     = (signal(t, T, z0, kappa) * rate['k1'] * (wc1_n**2) / (rate['K'] + (wc1_n**2))) - (rate['k4'] * frq_mrna) 
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


def border_behavior(z_strength, z_per, strain = rate, tcycle = 80, kappa = 0.5, signal = ztan):
    
    
    ### simulate arnold tongue
            

    z0= z_strength
    T = z_per

            ### define t so that there are enough temp cycles to cut out transients
            ### considering highest tau value
    t       = np.arange(0,tcycle*T,0.1)        
    state   = odeint(clock,state0,t,args=(strain,T,z0,kappa,signal))
                  
            ### find time after 85 temp cycles (time res. is 0.1)
            ### then get system state x0 after 85 temp cycles

            ### do the same for extrinsic zeitgeber function
    lt = int(-(25*10*T))     
            
    x0      = state[lt:,1]            
    tn      = t[lt:]
           
            ### get extrema and neighbors for zeitgeber function and simulated data
    frq_per  = get_periods(x0, tn)       
    period = np.mean(frq_per)

    print period        
            
            ### define entrainment criteria
            ### T-tau should be < 5 minutes
            ### period std should be small

    z_minima=np.arange(T/2,tcycle*T,T)
                #ph = get_phase(x0,tn,z0,tn)
                ### normalize phase to 2pi and set entr to that phase
    a = get_xmax(state[:,1],t)[-15:-1]
    b = z_minima[-15:-1]
    
    c = a-b
    
    
    c[c<0]=T+c[c<0]
    
    
    phase = 2*np.pi*c/T
    

       
    save_to = '/home/burt/neurospora/figures/desync/'
    
    fig, axes = plt.subplots(2,1,figsize=(12,9))
    axes = axes.flatten()
    ax = axes[0]
    ax1 = axes[1]
    
    ax.plot(tn,x0,"k")
    ax1.plot(tn, ztan(tn,T,z0,0.5),"k")
    
    #fig.savefig(save_to+"border"+"T_"+str(T)+".png")
    #plt.close(fig)
    

#zstr = 0.05
#T = 27

#border_behavior(zstr,20.0)

def border2(zstr,T):

    
    t       = np.arange(0,5000,.1)        
    state   = odeint(clock,state0,t,args=(rate,T,zstr,.5,ztan))
    z = ztan(t,T,zstr,.5)
    trans = 30000

    t       = t[-trans:]

    frq = state[-trans:,1]
    frq_mean = np.mean(frq)
    frq_detrend = frq-frq_mean

    z = z[-trans:]
    z_mean = np.mean(z)
    z_detrend = z-z_mean

    border_cut = 10000

    hann = np.hanning(len(frq_detrend))
    hamm = np.hamming(len(frq_detrend))
    black= np.blackman(len(frq_detrend))

    frq_signal = hilbert(hamm*frq_detrend)
    frq_signal = frq_signal[border_cut:-border_cut]
    frq_envelope = np.abs(frq_signal)
    frq_phase = np.angle(frq_signal)

    z_signal = hilbert(hamm*z_detrend)
    z_signal = z_signal[border_cut:-border_cut]
    z_envelope = np.abs(z_signal)
    z_phase = np.angle(z_signal)

    phase_diff = np.arctan2(np.sin(z_phase-frq_phase),np.cos(z_phase-frq_phase))
    phase_diff_max_idx = argrelextrema(phase_diff, np.greater)[0]
    phase_diff_max = phase_diff[phase_diff_max_idx]

    t_phase = phase_diff_max_idx/10

    mask = np.ma.array(phase_diff_max)

    mask[phase_diff_max>3.05] = np.ma.masked

    tn = t[border_cut:-border_cut]
    
    """
    fig = plt.figure()
    ax0 = fig.add_subplot(321)
    ax0.plot(t, hamm*frq_detrend, label='signal')
    ax0.plot(tn, frq_envelope, label='envelope')

    ax0.legend()
    ax1 = fig.add_subplot(322)
    ax1.plot(tn, frq_phase)

    ax3 = fig.add_subplot(323)
    ax3.plot(t, z_detrend, label='z signal')
    ax3.plot(tn, z_envelope, label='z envelope')

    ax3.legend()
    ax4 = fig.add_subplot(324)
    ax4.plot(tn, z_phase)


    ax5 = fig.add_subplot(325)
    ax5.scatter(tn,phase_diff,c = "k",s=.1)
    ax5.set_ylim(-4,4)
    """


    N = len(frq_detrend)/2+1
    X = np.linspace(0, 5, N, endpoint=True)

    Y = np.fft.fft(frq_detrend*hamm)

    xn=X[X>0]
    xn = 1/xn

    yn=2.0*np.abs(Y[1:N])/N
    power = 2.*np.abs(Y[1:N])**2/N
    norm = np.max(power)
    
    fig = plt.figure()
    ax = fig.add_subplot(311)

    ax.plot(xn, yn)
    ax.set_xlim(5,50)
    ax.set_title("T = "+str(T)+" , z = "+str(zstr))
    
    ax2 = fig.add_subplot(312)
    ax2.plot(xn,power/norm)
    ax2.set_xlim(5,50)
    
    ax3 = fig.add_subplot(313)
    ax3.scatter(tn,phase_diff,c = "k",s=.1)
    ax3.set_ylim(-4,4)
    
"""
duration = 1.0
fs = 400.0
samples = int(fs*duration)
t = np.arange(samples) / fs



signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)


fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(t, signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1 = fig.add_subplot(212)
ax1.plot(t[1:], instantaneous_frequency)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 120.0)
"""
#### make fourier analysis






#border_behavior(zstr,20.0)
"""


    
t       = np.arange(0,2000,.1)        
state   = odeint(clock,state0,t,args=(rate,T,zstr,.5,ztan))

frq = state[-18000:,1]
frq_mean = np.mean(frq)

frq_detrend = frq-frq_mean

hann = np.hanning(len(frq_detrend))
hamm = np.hamming(len(frq_detrend))
black= np.blackman(len(frq_detrend))

frq_signal = hilbert(hann*frq_detrend)
frq_envelope = np.abs(frq_signal)

tn = t[-18000:]
fig = plt.figure()
ax0 = fig.add_subplot(111)
ax0.plot(tn, frq_signal, label='signal')
ax0.plot(tn, frq_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
"""