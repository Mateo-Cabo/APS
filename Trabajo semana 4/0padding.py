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
N2 = 10000

# f0 = 1

SNR = 10 #db

n_pruebas = 200

omega0 = fs/4 #mitad de banda

a1 = np.sqrt(2)

pot_ruido_analog = 10**(-SNR/10)

ver = -10 * np.log(pot_ruido_analog)

#%% Señal

ts=1/fs
df=fs/N
df2 = df/10
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

# #%% Ploteo
# plt.figure(1)
# plt.plot(tt, sr)
# plt.show()


# plt.figure(2)
ft_As = 1/N * np.fft.fft(sr, axis=0, n = N2)
ff = np.linspace(0, (N-1)*df, N) # grilla de sampleo frecuencial
# bfrec = ff <= fs/2

# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), label='$ s $ (sig.)', marker = 'x' )
# plt.show()

#%% Ventanas a usar

#Blackman
#Flatop
#no hacer nada --> Rectangular
#Alguna otra que quiera

window1 = scipy.signal.windows.blackmanharris(N).reshape((N,1))
window2 = scipy.signal.windows.flattop(N).reshape((N,1))
window3 = scipy.signal.windows.bohman(N).reshape((N,1))

sw1 = window1 * sr
sw2 = window2 * sr
sw3 = window3 * sr

# plt.figure(3)

Sn1 =  1/N * abs(np.fft.fft(sw1, axis=0, n = N2))
Sn2 =  1/N * abs(np.fft.fft(sw2, axis=0, n = N2))
Sn3 =  1/N * abs(np.fft.fft(sw3, axis=0, n = N2))

# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(Sn1[bfrec])**2), label='$ s $ (sig.)', marker = 'x' )
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(Sn2[bfrec])**2), label='$ s $ (sig.)', marker = 'x' )
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(Sn3[bfrec])**2), label='$ s $ (sig.)', marker = 'x' )

# plt.show()

# a1_1 = Sn1[250,:]
# a1_2 = Sn2[250,:]
# a1_3 = Sn3[250,:]
# a1_4 = ft_As[250,:]

# plt.figure(4)

# plt.hist(abs(a1_1), bins = 10, label = 'BlackMan', alpha =  0.5)
# plt.hist(abs(a1_2), bins = 10, label = 'Flattop', alpha =  0.5)
# plt.hist(abs(a1_3), bins = 10, label = 'Bohman', alpha =  0.5)
# plt.hist(abs(a1_4), bins = 10, label = 'Rectangular', alpha =  0.5)
# plt.legend()

# plt.show()

#%% Estimador omega

k1 = np.argmax(Sn1[:N2//2:], axis = 0)
omega1_est = (k1) * df2

k2 = np.argmax(Sn2[:N2//2:], axis = 0)
omega2_est = (k2) * df2

k3 = np.argmax(Sn3[:N2//2:], axis = 0)
omega3_est = (k3) * df2

k4 = np.argmax(ft_As[:N2//2:], axis = 0)
omega4_est = (k4) * df2

plt.figure(5)

plt.hist(omega1_est, bins = 30, label = 'BlackMan', alpha =  0.5)
plt.hist(omega2_est, bins = 30, label = 'Flattop', alpha =  0.5)
plt.hist(omega3_est, bins = 30, label = 'Bohman', alpha =  0.5)
plt.hist(omega4_est, bins = 30, label = 'Rectangular', alpha =  0.5)
plt.legend()

plt.show()
