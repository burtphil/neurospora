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
import matplotlib.collections as collections


def ztan(t,T,z0,kappa, s=1):
    pi = np.pi
    om = 2*pi/T
    mu = pi/(om*np.sin(kappa*pi))
    cos1 = np.cos(om*t)
    cos2 = np.cos(kappa*pi)
    out = 1+2*z0*((1/pi)*np.arctan(s*mu*(cos1-cos2)))
    return out

### define functions
def z(t,T,z0, kappa):
    out = 1 + z0*np.cos(2*np.pi*t/T)
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


def tongue(z_strength, z_per, strain = rate, upper = 1.0, lower = 1.0, phase = 0, tcycle = 100, kappa = 0.5, signal = ztan):
    
    ratio = upper/lower
    zeit_mesh,warm_mesh = np.meshgrid(z_strength,z_per)
    entrain_mesh = np.zeros_like(zeit_mesh)

    ### simulate arnold tongue
    for idx, valx in enumerate(z_strength):
        for idy, valy in enumerate(z_per):
            

            z0= z_strength[idx]
            T = z_per[idy]

            ### define t so that there are enough temp cycles to cut out transients
            ### considering highest tau value
            t       = np.arange(0,tcycle*T,0.1)        
            state   = odeint(clock,state0,t,args=(strain,T,z0,kappa,signal))
                  
            ### find time after 85 temp cycles (time res. is 0.1)
            ### then get system state x0 after 85 temp cycles

            ### do the same for extrinsic zeitgeber function
            lt = int(-(15*10*T))     
            
            x0      = state[lt:,1]            
            tn      = t[lt:]
           
            ### get extrema and neighbors for zeitgeber function and simulated data
            frq_per  = get_periods(x0, tn)       
            period = np.mean(frq_per)        
            entr = 100
            
            ### define entrainment criteria
            ### T-tau should be < 5 minutes
            ### period std should be small

            c = np.abs(T-(ratio*period))*60
            c2 = np.std((np.diff(frq_per)*ratio)/T)            
            
            #print "per = "+str(round(period,3))+" , T = "+str(round(T,3))+" , z0 = "+str(round(z0,3))
            #print "c = "+str(c)+" , c2 = "+str(c2)
            
            if c < 5 and c2 < 0.5:
                
                print "entrained for "+"T= "+str(round(T,3))+" and z0= "+str(round(z0,3))
                
                if phase == 0:
                    entr = 0
                
                else:
                    z_minima=np.arange(T/2,tcycle*T,T)
                #ph = get_phase(x0,tn,z0,tn)
                ### normalize phase to 2pi and set entr to that phase
                    a = get_xmax(x0,tn)[-5:]
                    b = z_minima[-5:]
    
                    c = a-b
    
                    if np.sum(c)>0:
                        ph=np.mean(c)
                    else:
                        a=a+T
                        ph=np.mean(a-b)
                    entr = 2*np.pi*ph/T
            
            entrain_mesh[idy,idx] = entr
            print ""
    
    ent = entrain_mesh[:-1,:-1]
    
    
    date = datetime.strftime(datetime.now(), '%Y_%m_%d_%H')
    
    t_name = '_tongue_'+str(int(upper))+'_'+str(int(lower))+"_z_"+str(round(z_strength[-1],1))
    t_mask = np.ma.masked_array(ent, ent == 100)
    
    if phase == 1: 
        t_name = t_name + "_ph"
        
    save_to = '/home/burt/neurospora/figures/entrainment/'
    
    
    np.savez(save_to+date+t_name, warm_mesh = warm_mesh, zeit_mesh = zeit_mesh, ent = ent)

        

    fig, ax = plt.subplots(figsize=(12,9))
    heatmap = ax.pcolormesh(warm_mesh,zeit_mesh, t_mask, cmap = cc.m_colorwheel, edgecolors = "none", vmin = 0, vmax = 2*np.pi)
 
    if signal == z:
        ax.set_title("squarewave")
    ax.set_xlabel("T [h]", fontsize = 22)
    ax.set_ylabel("Z [a.u.]", fontsize = 22)
    ax.tick_params(labelsize = 16)
    
    
    if phase == 1:
        cbar = fig.colorbar(heatmap,ticks=[0,np.pi/2,np.pi,1.5*np.pi,2*np.pi], label = 'Phase (rad)')
        cbar.ax.set_yticklabels(['0','$\pi/2$','$\pi$','$3\pi/2$', '2$\pi$'])
        cbar.ax.tick_params(labelsize = 16)
        cbar.ax.set_ylabel("Phase (rad)", fontsize = 22)
        

    plt.tight_layout()
    fig.savefig(save_to+date+t_name+".pdf", dpi=1200)
    plt.show()
"""
def entr(state, t, T,z0,kappa,epsilon = 0.05):
    
    zt = ztan(t,T,z0,kappa)
    ### reshape zt
    zt = zt.reshape((zt.size,1))
    state = np.concatenate((state,zt), axis = 1)
                     ### variable for epsilon ball criterion
    t_state = int(85 * 10 * T)                   ### time after 85 temp cycles (time res. is 0.1)

    x0 = state[t_state,:]                   ### system at t state x0
    n_state = state[t_state:,:]            ### new array beginning at t state
                                            ### create empty list store t values that mach ball criterion
    t_new = t[t_state:]
    t_new = t_new.reshape((t_new.size,1))
    test = np.concatenate((n_state,t_new), axis = 1)
    a = np.linalg.norm(n_state-x0, axis = 1)
    entrain = test[a<epsilon]
    
    time = entrain[:,8]
    
    l = []
    
    for idx,x in enumerate(time[5:-1]):
        if np.abs(time[idx]-time[idx+1]) > 1:
            l.append(x)

    periods = np.diff(l)
    
    return np.std(periods/T)

"""
def zoom(T,z0,kappa, strain =rate, ratio = 1, tcycle = 100):

    #warm_dur = T*(1-kappa)
    t      = np.arange(0,tcycle*T,0.1)
    state = odeint(clock,state0,t,args=(strain,T,z0,kappa))
    

    lt = int(-(10*10*T))                
    x0      = state[lt:,1]            
    tn      = t[lt:]
    
    
    frq_per  = get_periods(x0, tn)       
    period = np.mean(frq_per)        
                
    ### define entrainment criteria
    ### T-tau should be < 5 minutes
    ### period std should be small
    c = np.abs(T-(ratio*period))*60
    #print c
    
    c2 = np.std((np.diff(frq_per)*ratio)/T)
    #print c2
    
    if c < (0.375*T) and c2 < 0.5:
        print "entrained, c ="+str(round(c,2))+" c2= "+str(round(c2,2))

        z_minima=np.arange(T/2,tcycle*T,T)
                #ph = get_phase(x0,tn,z0,tn)
                ### normalize phase to 2pi and set entr to that phase
        a = get_xmax(x0,tn)[-5:]
        b = z_minima[-5:]   
        c = a-b
        
        
        if np.sum(c)>0:
            ph=np.mean(c)
        else:
            a=a+T
            ph=np.mean(a-b)
        phase = 2*np.pi*ph/T
        
        print "phase = " + str(round(phase,1))

    
    fig, axes = plt.subplots(3,1,figsize= (12,9))
    axes = axes.flatten()
    ax = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]
    
    ax.plot(tn,x0, "k")
    ax.set_xlim(tn[0], tn[-1])
    ax.set_xticks(np.arange(tn[int(10*T*kappa/2)], tn[-1], T))
    collection = collections.BrokenBarHCollection.span_where(
        tn, ymin=100, ymax=-100, where= ztan(tn,T,z0, kappa) < 1, facecolor='gray', alpha=0.5)
    ax.add_collection(collection)
    
    ax1.plot(tn,ztan(tn,T,z0, kappa), "k")
    ax1.set_xlim(tn[0], tn[-1])
    ax1.set_xticks(np.arange(tn[int(10*T*kappa)], tn[-1], T/2))
    
    ax2.plot(state[lt:,0],state[lt:,1])
    


 #   col = collection
  #  ax1.add_collection(col)
"""
def newtongue(z_strength, z_per, strain = rate, upper = 1.0, lower = 1.0, phase = 0, tcycle = 100, kappa = 0.5, signal = ztan):
    
    #ratio = upper/lower
    zeit_mesh,warm_mesh = np.meshgrid(z_strength,z_per)
    entrain_mesh = np.zeros_like(zeit_mesh)

    ### simulate arnold tongue
    for idx, valx in enumerate(z_strength):
        for idy, valy in enumerate(z_per):
            

            z0= z_strength[idx]
            T = z_per[idy]

            ### define t so that there are enough temp cycles to cut out transients
            ### considering highest tau value
            t       = np.arange(0,tcycle*T,0.1)        
            state   = odeint(clock,state0,t,args=(strain,T,z0,kappa,signal))
                  
            ### find time after 85 temp cycles (time res. is 0.1)
            ### then get system state x0 after 85 temp cycles

            ### do the same for extrinsic zeitgeber function
            lt = int(-(15*10*T))     
            
            x0      = state[lt:,1]            
            tn      = t[lt:]
           
            ### get extrema and neighbors for zeitgeber function and simulated data
            #frq_per  = get_periods(x0, tn)       
            #period = np.mean(frq_per)        
            entrain = 0
            
            ### define entrainment criteria
            ### T-tau should be < 5 minutes
            ### period std should be small

            #c = np.abs(T-(ratio*period))*60
            #c2 = np.std(np.diff(frq_per))
            
            crit = entr(state,t,T,z0,kappa, 0.1)
            
            if crit < 0.05:
                print "entrained for "+"T= "+str(round(T,2))+" and z0= "+str(round(z0,2))
                
                if phase == 0:
                    entrain = 0
                
                else:
                    z_minima=np.arange(T/2,tcycle*T,T)
                #ph = get_phase(x0,tn,z0,tn)
                ### normalize phase to 2pi and set entr to that phase
                    a = get_xmin(x0,tn)[-5:]
                    b = z_minima[-5:]
    
                    c = a-b
    
                    if np.sum(c)>0:
                        ph=np.mean(c)
                    else:
                        a=a+T
                        ph=np.mean(a-b)
                    entrain = 2*np.pi*ph/T
            
            entrain_mesh[idy,idx] = entrain
    
    
    ent = entrain_mesh[:-1,:-1]
    
    t_mask = np.ma.masked_array(ent, ent == np.nan)
    date = datetime.strftime(datetime.now(), '%Y_%m_%d_%H')
    
    t_name = '_tonguenew_'+str(int(upper))+'_'+str(int(lower))+"_z_"+str(round(z_strength[-1],1))
    
    if phase == 1: 
        t_name = t_name + "_ph"
        
    save_to = '/home/burt/neurospora/figures/entrainment/'
    
    
    np.savez(save_to+date+t_name, warm_mesh = warm_mesh, zeit_mesh = zeit_mesh, ent = ent)

        

    fig, ax = plt.subplots(figsize=(12,9))
    heatmap = ax.pcolormesh(warm_mesh,zeit_mesh, t_mask, cmap = cc.m_colorwheel, edgecolors = "none", vmin = 0, vmax = 2*np.pi)
 
    if signal == z:
        ax.set_title("squarewave")
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
""" 
#tongue(np.linspace(0.001,0.1,100), np.linspace(19,25,100),upper = 1.0, lower = 1, phase = 1)
tongue(np.linspace(0.001,0.1,100), np.linspace(24,32,100),strain = rate7, upper = 1.0, lower = 1, phase = 1)
tongue(np.linspace(0.001,0.1,100), np.linspace(13,19,100),strain = rate1, upper = 1.0, lower = 1, phase = 1)
tongue(np.linspace(0.01,1,100), np.linspace(12,65,100),strain = rate1, upper = 1.0, lower = 1, phase = 1)
#newtongue(np.linspace(0.001,0.1,100), np.linspace(21,25,100), phase = 1)