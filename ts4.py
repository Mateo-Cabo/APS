# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 20:25:47 2025

@author: mateo
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy

#%% Variables

fs = 1000
N = 1000

# f0 = 1

SNR = 10 #db

n_pruebas = 10

omega0 = fs/4 #mitad de banda

a1 = 1/np.sqrt(2)

pot_ruido_analog = 10**(-SNR/10)

ver = -10 * np.log(pot_ruido_analog)

#%% Señal

ts=1/fs
df=fs/N
# nn = fs/f0
    
## Tiempo ##

# tt = np.arange(0,1,1/1000).reshape((1000,1))
# tt_m = np.repeat(tt, 200, axis = 1)

tt = np.linspace(0, (N-1)*ts, N).reshape((N,1))
tt = np.tile(tt, n_pruebas)

fr = np.random.uniform(-1/2, 1/2, size = (1,n_pruebas))

omega1 = omega0 + fr*(df)

## Argumento ##

arg = (omega1 * tt)

xx = a1*np.sin(2*np.pi*omega1*tt)

#%% Normalizo

xn = xx/np.std(xx)

nn =  np.random.normal(0, np.sqrt(pot_ruido_analog), N).reshape((N,1)) # señal de ruido de analógico

analog_sig = xn # señal analógica sin ruido

sr = analog_sig + nn # señal analógica de entrada al ADC (con ruido analógico)

# np.hstack 
# np.vstack