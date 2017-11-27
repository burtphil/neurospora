# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:46:21 2017

@author: Philipp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.collections as collections
import string


pi = np.pi

def ztan(t,T,kappa,z0, s=10):
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

def clock(state, t, rate,T,kappa,z0):
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
        
        dt_frq_mrna     = (ztan(t,T,kappa,z0)*rate['k1'] * (wc1_n**2) / (rate['K'] + (wc1_n**2))) - (rate['k4'] * frq_mrna) 
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

frq_mrna0    = 4.5
frq_c0       = 25
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

state1 = [4.7,
          25,
          frq_n0,
          wc1_mrna0,
          wc1_c0,
          wc1_n0,
          frq_n_wc1_n0]



    
    
### simulation parameters
Tcycle= np.array([14.29,14.299,25.333,25.334])


z0 = np.array([.5])
z1 = np.array([.1])

kappa= np.array([.5])

save_to = "/home/burt/figures/transients2/"


t_list =[]
FRQ_list = []
FRQ_last = []
FRQ_first = []
t_last = []
t_first = []
day_night_first = []
day_night_end = []
#### plot poincare sections for varying T and kappa


for T in Tcycle:
    
    t = np.arange(0,150*T,0.1)
    
    if (T == 25.333 or T == 25.334):
         z0 = .1
         t = np.arange(0,126*T,0.1)
         
    
    t_l = t[int(-(50*T)):]
    t_f =t[:int((50*T))]
    
    
    t_list.append(t)
    t_last.append(t_l)
    t_first.append(t_f)
    ### make arrays containing only the last part of simulation
    ### run simulation
    state = odeint(clock,state0,t,args=(rate,T,kappa,z0))  
    
    FRQ = state[:,1]
    FRQ_l = state[int(-(50*T)):,1]
    FRQ_f = state[:int((50*T)),1]
    
    FRQ_list.append(FRQ)
    FRQ_last.append(FRQ_l)
    FRQ_first.append(FRQ_f)
    
    warm_dur = kappa * T
    
    day_night_f = collections.BrokenBarHCollection.span_where(
        t_f, ymin=100, ymax=-100, where= ((t_f % T) <= warm_dur), facecolor='gray', alpha=0.5)
    
    day_night_l = collections.BrokenBarHCollection.span_where(
        t_l, ymin=100, ymax=-100, where= ((t_l % T) <= warm_dur), facecolor='gray', alpha=0.5)
    
    day_night_first.append(day_night_f)
    day_night_end.append(day_night_l)

def plotstyle(ax,t,T, ticks = 1):
    ax.set_xlim(t[0], t[-1])
    ax.set_xticks(np.arange(int(t[0]), int(t[-1]),ticks * T))

def ylabels(axes):
    for ax in axes:
        ax.set_ylabel("$FRQ_c$ (a.u.)", fontsize = 'xx-large')

def xlabels(axes):
    for ax in axes:
        ax.set_xlabel("time (h)", fontsize = 'xx-large')

def ticks(axes):
    for ax in axes:
        ax.tick_params(labelsize= 'x-large')
            
fig,axes = plt.subplots(4,3, figsize=(12,9))

axes = axes.flatten()
ax=axes[0]
ax1=axes[1]
ax2=axes[2]
ax3=axes[3]
ax4=axes[4]
ax5=axes[5]
ax6=axes[6]
ax7=axes[7]
ax8=axes[8]
ax9=axes[9]
ax10=axes[10]
ax11=axes[11]


ax.plot(t_first[0],FRQ_first[0],'k')
ax.add_collection(day_night_first[0])
ax.set_ylabel("$FRQ_c$ (a.u.)", fontsize = 'xx-large')
ax.set_title("First five T-cycles", fontsize = 'xx-large')
plotstyle(ax,t_first[0],Tcycle[0])
ax1.plot(t_list[0],FRQ_list[0],'k')
ax1.set_title("Whole simulation", fontsize = 'xx-large')
ax2.plot(t_last[0],FRQ_last[0],'k')
ax2.set_title("Last five T-cycles", fontsize = 'xx-large')
ax2.add_collection(day_night_end[0])
plotstyle(ax2,t_last[0],Tcycle[0], ticks = 2)

ax3.plot(t_first[1],FRQ_first[1],'k')
ax3.add_collection(day_night_first[1])
plotstyle(ax3,t_first[1],Tcycle[1])
ax4.plot(t_list[1],FRQ_list[1],'k')
ax5.plot(t_last[1],FRQ_last[1],'k')
ax5.add_collection(day_night_end[1])
plotstyle(ax5,t_last[1],Tcycle[1], ticks = 2)

ax6.plot(t_first[2],FRQ_first[2],'k')
ax6.add_collection(day_night_first[2])
plotstyle(ax6,t_first[2],Tcycle[2])
ax7.plot(t_list[2],FRQ_list[2],'k')
ax8.plot(t_last[2],FRQ_last[2],'k')
ax8.add_collection(day_night_end[2])
plotstyle(ax8,t_last[2],Tcycle[2], ticks = 2)

ax9.plot(t_first[3],FRQ_first[3],'k')
ax9.add_collection(day_night_first[3])
plotstyle(ax9,t_first[3],Tcycle[3])
ax10.plot(t_list[3],FRQ_list[3],'k')
ax11.plot(t_last[3],FRQ_last[3],'k')
ax11.add_collection(day_night_end[3])
ax11.set_yticks([20,30])
plotstyle(ax11,t_last[3],Tcycle[3], ticks = 2)

for n, ax in enumerate(axes): 
    ax.text(-0.15, .97, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=20, weight='bold')

ylabels([ax3,ax6,ax9])
xlabels([ax9,ax10,ax11])
ticks(axes)
plt.tight_layout()
fig.savefig("transients.pdf",dpi=1200)
plt.show()
"""
def plotstyle(ax,xlabel,ylabel):
    ax.set_xlabel(xlabel,fontsize= 'xx-large')
    ax.set_ylabel(ylabel,fontsize= 'xx-large')
    ax.tick_params(labelsize = 'x-large')
    
trace = state[int(-(50*T)):,1]
t = t[int(-(50*T)):]
ax.set_xlim(t[0], t[-1])
ax.set_xticks(np.arange(int(t[0]), int(t[-1]), T/2))
#ax2.set_xticks(np.arange(0, 1200, 12.0))

collection = collections.BrokenBarHCollection.span_where(
    t, ymin=100, ymax=-100, where= ((t % T) <= warm_dur), facecolor='gray', alpha=0.5)

ax.add_collection(collection)
"""

"""
            warm_dur = kappa*T
            
            fig,ax = plt.subplots(figsize=(24,9))

            title = "T= "+str(T)+" kappa= "+str(kappa)+" z0= "+str(z0)
            ax.plot(t,FRQ,"k")
            ax.set_xlim(t[0], t[-1])
            ax.set_xticks(np.arange(int(t[0]), int(t[-1]), T/2))
            #ax2.set_xticks(np.arange(0, 1200, 12.0))
            #collection = collections.BrokenBarHCollection.span_where(
            #t, ymin=100, ymax=-100, where= ((t % T) <= warm_dur), facecolor='gray', alpha=0.5)
            #ax.add_collection(collection)
            ax.tick_params(labelsize = 'x-large')

            #fig.savefig(save_to+"k_"+str(kappa)+"z_"+str(z0)+"T_"+str(T)+".png")
            plt.close(fig)
"""
"""
for T in Tcycle1:
     for z0 in zeitgeber_strength1:
            
            t = np.arange(0,240*T,0.1)
            t_l = t[int(-(50*T)):]
            t_f =t[:int((50*T))]
            
            
            t_list.append(t)
            t_last.append(t_l)
            t_first.append(t_f)
            ### make arrays containing only the last part of simulation
            ### run simulation
            state = odeint(clock,state0,t,args=(rate,T,kappa,z0))  
            

            FRQ = state[:,1]
            FRQ_l = state[int(-(50*T)):,1]
            FRQ_f = state[:int((50*T)),1]
            
            FRQ_list.append(FRQ)
            FRQ_last.append(FRQ_l)
            FRQ_first.append(FRQ_f)
"""
