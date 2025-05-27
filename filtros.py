# -*- coding: utf-8 -*-
"""
Created on Thu May 22 18:20:38 2025

@author: mateo
"""

#import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

from pytc2.sistemas_lineales import plot_plantilla

# Tipo de aproximación
        
aprox_name = 'butter'
# aprox_name = 'cheby1'
# aprox_name = 'cheby2'
# aprox_name = 'ellip'

fs = 1000 # Hz
nyq_frec= fs/2
fpass = np.array( [1.0, 35.0] ) 
ripple = 1.0 # dB #gpass
fstop = np.array( [0.1, 50.0] ) #gstop
attenuation = 40 # dB

# Diseñamos el filtro
sos = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype = aprox_name, output='SOS', fs=fs)

#Plantilla de diseño/analisis

npoints = 1000

w_rad = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi #muestreo en puntos donde yo quiera. Ya no es equiespaciado. Es una plantilla logaritmica

w, hh = sig.sosfreqz(sos, worN=w_rad)

plt.plot(w/np.pi*nyq_frec, 20*np.log10(np.abs(hh)+1e-15), label='sos')

# filtro anterior, como referencia
#w, mag, _ = my_digital_filter.bode(npoints)
#plt.plot(w/w_nyq, mag, label=my_digital_filter_desc)

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

ax = plt.gca()
#ax.set_xlim([0, 1])
#ax.set_ylim([-60, 1])

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
plt.legend()
plt.show()