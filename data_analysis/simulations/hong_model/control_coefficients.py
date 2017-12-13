#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:00:15 2017

@author: burt
"""

#### define all dictionaries used


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
from scipy.signal import argrelextrema


### define model variable names used in dictionaries

state_names = ["frq mRNA","FRQc","FRQn","wc-1 mRNA","WC-1c","WC-1n",
               "FRQn:WC-1n"]

phase_names = ["phase frq mRNA","phase FRQc","phase FRQn","phase wc-1 mRNA",
               "phase WC-1c","phase WC-1n","phase FRQn:WC-1n"]

amp_names = ["amplitude frq mRNA","amplitude FRQc","amplitude FRQn",
             "amplitude wc-1 mRNA","amplitude WC-1c","amplitude WC-1n"
             ,"amplitude FRQn:WC-1n"]

per_names = ["period frq mRNA","period FRQc","period FRQn","period wc-1 mRNA",
             "period WC-1c","period WC-1n","period FRQn:WC-1n"]

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



#### functions
##############################################################################
##############################################################################
##############################################################################

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

def get_per(arr):
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
    ph = 2*np.pi*c
    
    return ph

def get_amp(ma,mi):
    amp = np.mean(ma)-np.mean(mi)
    amp = amp/2
    return amp

def write_var(state):
    get_amp(state)
    get_per(state)
    get_phase(state)
    

def make_amp_dict(state,ref_state,p,pref):
    """
    Take state variable from ode simulation
    Calls amplitude function
    Returns dictionary of amplitudes
    """
    amp_dict = {}
    no_trans = remove_trans(state)
    
    for idx, valx in enumerate(amp_names):
        current_state = no_trans[:,idx]
        ma = get_extrema(current_state,ref_t)[0]
        mi = get_extrema(current_state,ref_t)[1]
        ma_ref= get_extrema(ref_state[idx], ref_t)[0]
        mi_ref= get_extrema(ref_state[idx],ref_t)[1]
        
        if get_amp(ref_state[:,idx]) != 0:
            rel_amp = rel_difference(get_amp(ma,mi), get_amp(ma_ref,mi_ref))/rel_difference(p,pref)
            amp_dict[valx] = rel_amp * 100
        else:

            amp_dict[valx] = 0
            
    return amp_dict    
    
def make_phase_dict(state,ref_state,p,pref):
    """
    Take state variable from ode simulation
    Calls phase function
    Returns dictionary of phases
    """    
    phase_dict = {}
    no_trans = remove_trans(state)
    frq_mrna_state = no_trans[:,0]
    
    for idx, valx in enumerate(phase_names):
        current_state = no_trans[:,idx]
        
        if get_phase(ref_state[:,idx],ref_state[:,0]) != 0:
            rel_phase = rel_difference(get_phase(current_state, frq_mrna_state),get_phase(ref_state[:,idx],ref_state[:,0]))/rel_difference(p,pref)
            phase_dict[valx]= rel_phase * 100
        else:    
            phase_dict[valx]= 0

    return phase_dict

def make_period_dict(state, ref_state,p,pref):
    """
    Take state variable from ode simulation
    Calls period function
    Returns dictionary of periods
    """
    period_dict = {}
    no_trans = remove_trans(state)  
    
    for idx, valx in enumerate(per_names):
        current_state = no_trans[:,idx]
        
        if get_per(ref_state[:,idx]) != 0:
            rel_per = rel_difference(get_per(current_state), get_per(ref_state[:,idx]))/rel_difference(p,pref)
            period_dict[valx] = rel_per * 100
        else:
            period_dict[valx] = 0
            
    return period_dict

"""
def get_period(current_state):   
    
    Take column of state variable from ode simulation
    calculate period by calculating distance between local maxima (see maxima_dist fct)
    Returns mean period of input variable
    
    period = maxima_dist(current_state) / 10.0
   # assert math.isnan(period) == False
    
    return period
"""
"""    
def get_amp(current_state):  
    
    Take column of state variable from ode simulation
    Calculate amplitude by subtracting local maxima from local minima and divide by two
    Returns mean of amplitudes of input variable
    
    amp = (get_maxima(current_state) - get_minima(current_state)) / 2
    #assert math.isnan(amp) == False
    
    return amp
"""
"""
def get_phase(current_state, frq_mrna_state):
    
    Take column of state variable from ode simulation
    Calculate phase by subtracting indices of local maxima from frq_mrna reference maxima indices
    Define phase to be always positive
    Normalize to 1 (input variables own period)
    Returns mean of phases of input variable
    
    maxima_idx = get_maxima_idx(current_state)
    first_maxima_idx = maxima_idx[:10]
    frq_mrna_maxima_idx = get_maxima_idx(frq_mrna_state)
    frq_mrna_maxima_idx = frq_mrna_maxima_idx[:10]
    
    if first_maxima_idx.any() :
        phase = first_maxima_idx - frq_mrna_maxima_idx   
        if np.sum(phase) < 0 :
            phase = frq_mrna_maxima_idx - first_maxima_idx
    
        relative_phase = np.mean(phase) / maxima_dist(current_state)
        phase = relative_phase
        assert relative_phase >= 0      
    else:
        phase = 0
        
    #assert math.isnan(phase) == False
    
    return phase
"""

def rel_difference(value, ref_value):
    """
    Return the relative difference
    """
    assert ref_value != 0
    rel_diff = np.abs(value-ref_value)/ref_value
    
    return rel_diff

    
def remove_trans(state):
    """
    Take state variable from ode simulation
    Remove transients from state variable
    Return state variable without transients
    """
    return np.array(state[1600:,:])

def clock(state, t, rate):

        ### purpose:simulate Hong et al 2008 model for neuropora clock
        """
        Take state variable from ode simulation
        Calls amplitude function
        Returns dictionary of amplitudes
        """

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
 
##############################################################################
##############################################################################
##############################################################################

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

t      = np.arange(0,480,0.1)

### what is a proper time resolution?

### run simulation
state = odeint(clock,state0,t,args=(rate,))  

state_names = ["frq mRNA","FRQc","FRQn","wc-1 mRNA","WC-1c","WC-1n","FRQn:WC-1n"]

var_dict_plus = rate.copy()
var_dict_minus = rate.copy()
amplitudes_pos = rate.copy()
amplitudes_neg = rate.copy()
periods_pos = rate.copy()
periods_neg = rate.copy()
phases_pos = rate.copy()
phases_neg = rate.copy()


### run a simulation varying each parameter of the model susequently by +- 10 percent
### initiate a reference state from the simulation with original parameters

ref_state = remove_trans(state)
ref_t = t[1600:]
### get amplitudes phases and period for ref_state



for key in rate:
    """
    for each parameter in model, change its value +-10% and calculate amplitudes,
    periods and phase for each model variable. store everything in a nested dict
    that contains the parameter as main key 
    """
    
    ### copy the original rate dictionary
    params_plus = rate.copy()
    params_minus = rate.copy()

    ### store current parameter and reference 
    pref = params_plus[key]
    pplus= pref * 1.1
    pminus= pref * 0.9

    #### change parameter values in copied dicts
    params_plus[key] = params_plus[key] * 1.1
    params_minus[key] = params_minus[key] * 0.9

       

    ### run simulation with new parameters
    state_plus = odeint(clock,state0,t,args=(params_plus,))
    state_minus = odeint(clock,state0,t,args=(params_minus,))
    
    ### get amplitude period and phase for each variable
    
    amp_dict_plus = make_amp_dict(state_plus, ref_state, pplus, pref)
    per_dict_plus = make_period_dict(state_plus, ref_state, pplus, pref)
    phase_dict_plus = make_phase_dict(state_plus, ref_state, pplus, pref)
    
    amp_dict_minus = make_amp_dict(state_minus, ref_state, pminus, pref)
    per_dict_minus = make_period_dict(state_minus, ref_state, pminus, pref)
    phase_dict_minus = make_phase_dict(state_minus, ref_state, pminus, pref)
    
    
    amplitudes_pos[key]=amp_dict_plus
    amplitudes_neg[key]=amp_dict_minus
    periods_pos[key]=per_dict_plus
    periods_neg[key]=per_dict_minus
    phases_pos[key]=phase_dict_plus
    phases_neg[key]=phase_dict_minus    
    #### combine amp phase and per into dictionary
    ### dict contains parameter as key and a dict with amp per and phase key value pairs
    var_dict_plus[key] = dict(amp_dict_plus.items() + per_dict_plus.items() + phase_dict_plus.items())
    var_dict_minus[key] = dict(amp_dict_minus.items() + per_dict_minus.items() + phase_dict_minus.items())

### convert dictionaries of parameter changes into pandas dfs
amplitudes_pos_df = pd.DataFrame.from_dict(amplitudes_pos, orient = "index")   
amplitudes_neg_df = pd.DataFrame.from_dict(amplitudes_neg, orient = "index")
periods_pos_df = pd.DataFrame.from_dict(periods_pos, orient = "index")   
periods_neg_df = pd.DataFrame.from_dict(periods_neg, orient = "index")
phases_pos_df = pd.DataFrame.from_dict(phases_pos, orient = "index")   
phases_neg_df = pd.DataFrame.from_dict(phases_neg, orient = "index")



var_plus_df = pd.DataFrame.from_dict(var_dict_plus, orient='index')
var_minus_df = pd.DataFrame.from_dict(var_dict_minus, orient='index')

### take column header names and add either + or -
var_plus_df_names = list(var_plus_df.columns.values)
var_minus_df_names = list(var_minus_df.columns.values)

for idx,n in enumerate(var_minus_df_names):   
    var_minus_df_names[idx] = n + " -"

for idx,n in enumerate(var_plus_df_names):   
    var_plus_df_names[idx] = n + " +"

    
var_plus_df.columns = var_plus_df_names
var_minus_df.columns = var_minus_df_names


### combine data frames
var_df = pd.concat([var_plus_df,var_minus_df], axis = 1)
"""
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('hong_robustness.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
var_df.to_excel(writer, sheet_name='Sheet1')

workbook  = writer.book
worksheet = writer.sheets['Sheet1']

# Light red fill with dark red text.
format_red = workbook.add_format({'bg_color': '#FFC7CE'})

format_dark_red = workbook.add_format({'bg_color': '#9C0006'})


# Add a format. Green fill with dark green text.
format_green = workbook.add_format({'bg_color': '#C6EFCE'})
    
format_dark_green = workbook.add_format({'bg_color': '#006100'})

    
worksheet.conditional_format('B2:AQ18', {'type':     'cell',
                                    'criteria': 'between',
                                    'minimum':      1,
                                    'maximum':      50,
                                    'format':    format_red})

worksheet.conditional_format('B2:AQ18', {'type':     'cell',
                                    'criteria': 'greater than',
                                    'value':     50,
                                    'format':    format_dark_red})

    
worksheet.conditional_format('B2:AQ18', {'type':     'cell',
                                    'criteria': 'between',
                                    'minimum':     -50,
                                    'maximum':      -1,
                                    'format':    format_green})

worksheet.conditional_format('B2:AQ18', {'type':     'cell',
                                    'criteria': 'less than',
                                    'value':     -50,
                                    'format':    format_dark_green})

    
workbook.close()
 """
##############################################################################
##############################################################################
##############################################################################

### plot bar charts



amplitudes_pos_df.plot.bar()
plt.ylabel("Relative amplitude change [%]")
plt.title("Parameter change plus 10 %")
plt.show()

amplitudes_neg_df.plot.bar()
plt.ylabel("Relative amplitude change [%]")
plt.title("Parameter change minus 10 %")
plt.show()

periods_pos_df.plot.bar()
plt.ylabel("Relative period change [%]")
plt.title("Parameter change plus 10 %")
plt.show()

periods_neg_df.plot.bar()
plt.ylabel("Relative period change [%]")
plt.title("Parameter change minus 10 %")
plt.show()

phases_pos_df.plot.bar()
plt.ylabel("Change in phase relative to frq mrna [%]")
plt.title("Parameter change plus 10 %")
plt.show()

phases_neg_df.plot.bar()
plt.ylabel("Change in phase relative to frq mrna [%]")
plt.title("Parameter change minus 10 %")
plt.show()

### plot the basic simulation
plt.figure()
plt.plot(t,state)
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.legend(state_names,loc='center left', bbox_to_anchor=(0.6, 0.5))
plt.show()

###
model_subplots = plt.figure(figsize=(8,12))
plt.subplot(4,2,1)
plt.plot(t, state[:,0])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.title('frq mRNA')

plt.subplot(4,2,2)
plt.plot(t, state[:,1])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.title('FRQc')

plt.subplot(4,2,3)
plt.plot(t, state[:,2])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.title('FRQn')

plt.subplot(4,2,4)
plt.plot(t, state[:,3])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.title('wc-1 mRNA')

plt.subplot(4,2,5)
plt.plot(t, state[:,4])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.title('WC-1c')

plt.subplot(4,2,6)
plt.plot(t, state[:,5])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.title('WC-1n')

plt.subplot(4,2,7)
plt.plot(t, state[:,6])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 49, 12.0))
plt.title('FRQn:WC-1n')
plt.tight_layout()

model_subplots.show()
