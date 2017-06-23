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
import pandas as pd

####################################################################
############# define variables and dictionaries #########################
####################################################################

### entrainment parameters
zeitgeber_period = 21.60
warm_dur = 17.33
temp_period = warm_dur / zeitgeber_period

run_time = 105 * zeitgeber_period
### other parameters

R = 8.314 # gas constant
ode_frq = 10 ### define frequency for odeint time resolution


### define functions
### temperature jumps between warm and cold cycle
def Temp(t):
    if ((t % zeitgeber_period) <= warm_dur) :
        return 301.15      
    else:
        return 295.15
    
#### linear arrhenius eqn for rate constants with temp dependency
def rate_k1(t):
    k = act_f['A1'] * np.exp(- act_e['E1']/(R * Temp(t)))
    return k

def rate_k2(t):
    k = act_f['A2'] * np.exp(- act_e['E2']/(R * Temp(t)))
    return k

def rate_k3(t):
    k = act_f['A3'] * np.exp(- act_e['E3']/(R * Temp(t)))
    return k

def rate_k4(t):
    k = act_f['A4'] * np.exp(- act_e['E4']/(R * Temp(t)))
    return k
### function for k5 which behaves not like arrhenius
def rate_k5(t):
    dummy = (0.00472727 * Temp(t))-1.1432546
    return dummy

def rate_k6(t):
    k = act_f['A6'] * np.exp(- act_e['E6']/(R * Temp(t)))
    return k

def rate_k7(t):
    k = act_f['A7'] * np.exp(- act_e['E7']/(R * Temp(t)))
    return k

def rate_k8(t):
    k = act_f['A8'] * np.exp(- act_e['E8']/(R * Temp(t)))
    return k

def rate_k9(t):
    k = act_f['A9'] * np.exp(- act_e['E9']/(R * Temp(t)))
    return k

def rate_k10(t):
    k = act_f['A10'] * np.exp(- act_e['E10']/(R * Temp(t)))
    return k

def rate_k11(t):
    k = act_f['A11'] * np.exp(- act_e['E11']/(R * Temp(t)))
    return k

def rate_k12(t):
    k = act_f['A12'] * np.exp(- act_e['E12']/(R * Temp(t)))
    return k

def rate_k13(t):
    k = act_f['A13'] * np.exp(- act_e['E13']/(R * Temp(t)))
    return k

def rate_k14(t):
    k = act_f['A14'] * np.exp(- act_e['E14']/(R * Temp(t)))
    return k

def rate_k15(t):
    k = act_f['A15'] * np.exp(- act_e['E15']/(R * Temp(t)))
    return k
### function for K which behaves not like arrhenius
def rate_K(t):
    if Temp(t) >= 297 :
        return 1.25
    else:
        return 0.02

def rate_K2(t):
    k = act_f['Ak2'] * np.exp(- act_e['Ek2']/(R * Temp(t)))
    return k
### function for k5 which does not behave like arrhenius

#### dictioniary of activation energies, k5 and K missing
act_e = {
        'E1'  :62.6,
        'E2'  :20.9,
        'E3'  :25.4,
        'E4'  :15.2,
        'E6'  :31.9,
        'E7'  :104,
        'E8'  :22,
        'E9'  :66.2,
        'E10' :30,
        'E11' :50.6,
        'E12' :25.4,
        'E13' :58.6,
        'E14' :50.2,
        'E15' :50.4,
        'Ek2' :68.8
}
    
 ### dicitinoary of prexop factors from arrhenius eqn, k5 and K missing
 
act_f = {
        'A1'  :1.84603598,
        'A2'  :1.81524074,
        'A3'  :0.05051497,
        'A4'  :0.23141468,
        'A6'  :0.07090665,
        'A7'  :0.52142402,
        'A8'  :0.80713176,
        'A9'  :41.0826429,
        'A10' :0.30365282,
        'A11' :0.05103114,
        'A12' :0.02020599,
        'A13' :51.1960968,
        'A14' :1.02045803,
        'A15' :8.16432297,
        'Ak2' :1.02814391
}

#### dictionary with all rate functions
factors = {
    'k1'    : rate_k1,
    'k2'    : rate_k2,
    'k3'    : rate_k3,
    'k4'    : rate_k4,
    'k5'    : rate_k5,
    'k6'    : rate_k6,
    'k7'    : rate_k7,
    'k8'    : rate_k8,
    'k9'    : rate_k9,
    'k10'   : rate_k10,
    'k11'   : rate_k11,
    'k12'   : rate_k12,
    'k13'   : rate_k13,
    'k14'   : rate_k14,
    'k15'   : rate_k15,
    'K'     : rate_K,
    'K2'    : rate_k2
}


params = {
    'rate_constants': factors
}

####################################################################
############# definde ODE system           #########################
####################################################################


### and a function of rate dependency on T: k(T(t))
def clock(state, t, params):
        ### purpose:simulate Hong et al 2008 model for neuropora clock


        ### define state vector

        frq_mrna    = state[0]
        frq_c       = state[1]
        frq_n       = state[2]
        wc1_mrna    = state[3]
        wc1_c       = state[4]
        wc1_n       = state[5]
        frq_n_wc1_n = state[6]
        
        rate = params['rate_constants']
        
        ###  ODEs Hong et al 2008
        ### letzter summand unklar bei dtfrqmrna
        dt_frq_mrna     = (rate['k1'](t) * (wc1_n**2) / (rate_K(t) + (wc1_n**2))) - (rate['k4'](t) * frq_mrna) 
        dt_frq_c        = (rate['k2'](t) * frq_mrna) - (rate['k3'](t) + rate['k5'](t) * frq_c)
        dt_frq_n        = (rate['k3'](t) * frq_c) + (rate['k14'](t) * frq_n_wc1_n) - (frq_n * (rate['k6'](t) + (rate['k13'](t) * wc1_n)))        
        dt_wc1_mrna     = rate['k7'](t) - (rate['k10'](t) * wc1_mrna)
        dt_wc1_c        = ((rate['k8'](t) * frq_c * wc1_mrna) / (rate['K2'](t) + frq_c)) - ((rate['k9'](t) + rate['k11'](t)) * wc1_c)
        dt_wc1_n        = (rate['k9'](t) * wc1_c) - (wc1_n * (rate['k12'](t) + (rate['k13'](t) * frq_n))) + (rate['k14'](t) * frq_n_wc1_n)
        dt_frq_n_wc1_n  = rate['k13'](t) * frq_n * wc1_n - ((rate['k14'](t) + rate['k15'](t)) * frq_n_wc1_n)
        
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
wc1_mrna0    = 1.66
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

t      = np.arange(0,run_time,0.1)

### what is a proper time resolution?

### run simulation
state = odeint(clock,state0,t,args=(params,))  



epsilon = 0.01                          ### variable for epsilon ball criterion
t_state = int(85 * 10 * zeitgeber_period)    ### time after 85 temp cycles (time res. is 0.1)

x0 = state[t_state,:]                   ### system at t state x0
n_state = state[t_state:,:]             ### new array beginning at t state
t_ball_pos = []                         ### create empty list store t values that mach ball criterion

### for loop that checks if system is similar to x0 
### computes the vector of the difference and checks if epsilon criterion is met

for i in n_state:
    diff = x0 - i
    if np.linalg.norm(diff) < epsilon:
        t_ball_pos.append(i)
        
t_ball_pos = np.array(t_ball_pos)

### make state to a pandas df

state_names = ['frq mRNA',
               'FRQc',
               'FRQn',
               'wc-1 mRNA',
               'WC-1c',
               'WC-1n',
               'FRQn:WC-1n']

### convert state array to data frame
df_state = pd.DataFrame(state, columns = state_names)

### add time column to data frame df_state
df_state['time'] = pd.Series(t)

### combine time data frame with t_ball crit pos data frame 
df_t_ball_pos = pd.DataFrame(t_ball_pos, columns = state_names)

df_merge = pd.merge(df_state, df_t_ball_pos)

### get only time column

times = df_merge['time']
### convert time column to array
times = np.array(times)
### get differences between t_n and t_n+1
times_diff = np.diff(times)

### calculate the mean of the differences and divide by zeitgeber period

times_mean = np.mean(times_diff)

### define final entrainment criterion
entrain_crit = times_mean / zeitgeber_period
entrain = 0
if entrain_crit < 1.1 and entrain_crit > 0.9:
    entrain = 1
else:
    entrain = 0

print "zeitgeber = " + str(zeitgeber_period)    
print "entrain = " + str(entrain)
print "n state = " + str(n_state.shape)
print "shape tball = " + str(t_ball_pos.shape)
print "entrain crit = " + str(entrain_crit)
### entrainment crit should be close to 1 if entrained
### next step: test entrainment for different thermoperiods and zetgeber periods







































