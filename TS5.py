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

n_pruebas = 200

omega0 = fs/4 #mitad de banda

a1 = np.sqrt(2)

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

nn =  np.random.normal(0, np.sqrt(pot_ruido_analog), (N,n_pruebas)) # señal de ruido de analógico

analog_sig = xn # señal analógica sin ruido

sr = analog_sig + nn # señal analógica de entrada al ADC (con ruido analógico)

# #%% Ploteo
plt.figure(1)
plt.plot(tt, sr)
plt.show()


plt.figure(2)
ft_As = 1/N * np.fft.fft(sr, axis=0)
ff = np.linspace(0, (N-1)*df, N) # grilla de sampleo frecuencial
bfrec = ff <= fs/2

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2))
plt.show()

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

plt.figure(3)

Sn1 =  1/N * np.fft.fft(sw1, axis=0)
Sn2 =  1/N * np.fft.fft(sw2, axis=0)
Sn3 =  1/N * np.fft.fft(sw3, axis=0)

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(Sn1[bfrec])**2), label='$ s $ (sig.)', marker = 'x' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(Sn2[bfrec])**2), label='$ s $ (sig.)', marker = 'x' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(Sn3[bfrec])**2), label='$ s $ (sig.)', marker = 'x' )

plt.show()

a1_1 = Sn1[250,:]
a1_2 = Sn2[250,:]
a1_3 = Sn3[250,:]
a1_4 = ft_As[250,:]

plt.figure(4)

plt.hist(abs(a1_1), bins = 10, label = 'BlackMan', alpha =  0.5)
plt.hist(abs(a1_2), bins = 10, label = 'Flattop', alpha =  0.5)
plt.hist(abs(a1_3), bins = 10, label = 'Bohman', alpha =  0.5)
plt.hist(abs(a1_4), bins = 10, label = 'Rectangular', alpha =  0.5)
plt.legend()

plt.show()
#%% welch

f, pxx = scipy.signal.welch(sr, fs=fs, window='hann', nperseg=N//4, noverlap=None, nfft=N, detrend='constant', return_onesided=True, scaling='density', axis=0, average='mean')

a2 = pxx[250,:]

plt.figure(7)
plt.plot(f,10*np.log(2*np.abs(pxx)**2))
plt.show()

plt.figure(6)

plt.hist(abs(a1_1), bins = 10, label = 'BlackMan', alpha =  0.5)
plt.hist(abs(a1_2), bins = 10, label = 'Flattop', alpha =  0.5)
plt.hist(abs(a1_3), bins = 10, label = 'Bohman', alpha =  0.5)
plt.hist(abs(a1_4), bins = 10, label = 'Rectangular', alpha =  0.5)
plt.hist(a2, bins = 10, label = 'Welch', alpha =  0.5)

plt.legend()

plt.show()

sesgo_w = np.mean(a2 - a1)
var_w = np.mean((a2 - a1)**2)

#%% Estimador omega

k1 = np.argmax(abs(Sn1[bfrec]), axis = 0)
omega1_est = np.abs(k1) * df

k2 = np.argmax(abs(Sn2[bfrec]), axis = 0)
omega2_est = np.abs(k2) * df

k3 = np.argmax(abs(Sn3[bfrec]), axis = 0)
omega3_est = np.abs(k3) * df

k4 = np.argmax(abs(ft_As[bfrec]), axis = 0)
omega4_est = np.abs(k4) * df

plt.figure(5)

plt.hist(omega1_est, bins = 10, label = 'BlackMan', alpha =  0.5)
plt.hist(omega2_est, bins = 10, label = 'Flattop', alpha =  0.5)
plt.hist(omega3_est, bins = 10, label = 'Bohman', alpha =  0.5)
plt.hist(omega4_est, bins = 10, label = 'Rectangular', alpha =  0.5)
plt.legend()

plt.show()

#%% Sesgo y varianza

sesgo1_a = np.mean(abs(a1_1 - a1))
sesgo2_a = np.mean(abs(a1_2 - a1))
sesgo3_a = np.mean(abs(a1_3 - a1))
sesgo4_a = np.mean(abs(a1_4 - a1))

var1_a = np.mean((abs(a1_1 - a1)**2))
var2_a = np.mean((abs(a1_2 - a1)**2))
var3_a = np.mean((abs(a1_3 - a1)**2))
var4_a = np.mean((abs(a1_4 - a1)**2))

sesgo1_o = np.mean(abs(omega1_est - omega1))
sesgo2_o = np.mean(abs(omega2_est - omega1))
sesgo3_o = np.mean(abs(omega3_est - omega1))
sesgo4_o = np.mean(abs(omega4_est - omega1))

var1_o = np.mean((abs(omega1_est - omega1)**2))
var2_o = np.mean((abs(omega2_est - omega1)**2))
var3_o = np.mean((abs(omega3_est - omega1)**2))
var4_o = np.mean((abs(omega4_est - omega1)**2))

#%% Tablas

print("Blackman Harris: ")
print("          ♦ Estimador a = ", np.mean(abs(a1_1)))
print("               Sesgo = ", sesgo1_a)
print("               Varianza = ", var1_a)
print("          ♦ Estimador omega1 = ", np.mean(abs(omega1_est)))
print("               Sesgo = ", sesgo1_o)
print("               Varianza = ", var1_o)

print("\nEstimadores Flattop: ")
print("          ♦ Estimador a = ", np.mean(abs(a1_2)))
print("               Sesgo = ", sesgo2_a)
print("               Varianza = ", var2_a)
print("          ♦ Estimador omega1 = ", np.mean(abs(omega2_est)))
print("               Sesgo = ", sesgo2_o)
print("               Varianza = ", var2_o)

print("\nEstimadores Bohman: ")
print("          ♦ Estimador a = ", np.mean(abs(a1_3)))
print("               Sesgo = ", sesgo3_a)
print("               Varianza = ", var3_a)
print("          ♦ Estimador omega1 = ", np.mean(abs(omega3_est)))
print("               Sesgo = ", sesgo3_o)
print("               Varianza = ", var3_o)

print("\nEstimadores Rectangular: ")
print("          ♦ Estimador a = ", np.mean(abs(a1_4)))
print("               Sesgo = ", sesgo4_a)
print("               Varianza = ", var4_a)
print("          ♦ Estimador omega1 = ", np.mean(abs(omega4_est)))
print("               Sesgo = ", sesgo4_o)
print("               Varianza = ", var4_o)



