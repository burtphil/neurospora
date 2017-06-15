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


### define variables

### entrainment parameters
zeitgeber_period = 40
warm_dur = 12
temp_period = warm_dur / zeitgeber_period

### other parameters

R = 8.314 # gas constant
t = 0 # time dummy variable


### define functions
### temperature jumps between warm and cold cycle
def Temp(t):
    if ((t % zeitgeber_period) <= warm_dur) :
        return 301.15      
    else:
        return 295.15
    
#### linear arrhenius eqn for rate constants with temp dependency
def rate_k(energy,a_factor,t):
    k = a_factor * np.exp(-energy/(Temp(t)*R))
    return k

### function for K which behaves not like arrhenius
def rate_K(t):
    if Temp(t) >= 297 :
        return 1.25
    else:
        return 0.02

### function for k5 which does not behave like arrhenius
def rate_k5(t):
    dummy = (0.00472727 * Temp(t))-1.1432546
    return dummy

#### dictioniary of activation energies
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
    
 ### dicitinoary of prexop factors from arrhenius eqn   
 
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


rate_constants = {
    'k1'    : rate_k(act_e['E1'],act_f['A1'],t),
    'k2'    : rate_k(act_e['E2'],act_f['A2'],t),
    'k3'    : rate_k(act_e['E3'],act_f['A3'],t),
    'k4'    : rate_k(act_e['E4'],act_f['A4'],t),
    'k5'    : rate_k5(t),
    'k6'    : rate_k(act_e['E6'],act_f['A6'],t),
    'k7'    : rate_k(act_e['E7'],act_f['A7'],t),
    'k8'    : rate_k(act_e['E8'],act_f['A8'],t),
    'k9'    : rate_k(act_e['E9'],act_f['A9'],t),
    'k10'   : rate_k(act_e['E10'],act_f['A10'],t),
    'k11'   : rate_k(act_e['E11'],act_f['A11'],t),
    'k12'   : rate_k(act_e['E12'],act_f['A12'],t),
    'k13'   : rate_k(act_e['E13'],act_f['A13'],t),
    'k14'   : rate_k(act_e['E14'],act_f['A14'],t),
    'k15'   : rate_k(act_e['E15'],act_f['A15'],t),
    'K'     : rate_K(t),
    'K2'    : rate_k(act_e['Ek2'],act_f['Ak2'],t)
}


params = {
    'rate_constants': rate_constants
}
### define ODE clock function

### write a function of Temperature depending on time: Temp(t)


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
wc1_mrna0    = rate_constants['k7']/rate_constants['k10']
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
state = odeint(clock,state0,t,args=(params,))  

### plot all ODEs
plt.plot(t,state)
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 481, 6))
plt.legend(["frq mRNA","FRQc","FRQn","wc-1 mRNA","WC-1c","WC-1n","FRQn:WC-1n"],loc='center left', bbox_to_anchor=(0.6, 0.5))
plt.show()


###
plt.figure(figsize=(8,12))
plt.subplot(4,2,1)
plt.plot(t, state[:,0])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 481, 6))
plt.title('frq mRNA')

plt.subplot(4,2,2)
plt.plot(t, state[:,1])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 481, 6))
plt.title('FRQc')

plt.subplot(4,2,3)
plt.plot(t, state[:,2])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 481, 6))
plt.title('FRQn')

plt.subplot(4,2,4)
plt.plot(t, state[:,3])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 481, 6))
plt.title('wc-1 mRNA')

plt.subplot(4,2,5)
plt.plot(t, state[:,4])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 481, 6))
plt.title('WC-1c')

plt.subplot(4,2,6)
plt.plot(t, state[:,5])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 481, 6))
plt.title('WC-1n')

plt.subplot(4,2,7)
plt.plot(t, state[:,6])
plt.xlabel("time [h]")
plt.ylabel("a.u")
plt.xticks(np.arange(0, 481, 6))
plt.title('FRQn:WC-1n')
plt.tight_layout()
plt.show()
