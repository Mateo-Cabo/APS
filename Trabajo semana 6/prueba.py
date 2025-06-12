# -*- coding: utf-8 -*-
"""
Created on Mon May 26 18:26:02 2025

@author: mateo
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import scipy.optimize


fs = 1000

w = 2*np.pi * fs

omega = np.arange(0, 2*np.pi, np.pi/4000)

arg = omega  * 1j

z = np.exp(arg)

H = z**0 + z**-1 + z**-2 + z**-3 + z**-4
H_mod = abs(H)
#H = (z**(3/2))*(z**(-3/2) + z**(-1/2) + z**(1/2) + z**(3/2))/(z**3)
#H = z**(-3/2)*(2*np.cos(3/2*omega) + 2*np.cos(1/2*omega))

ceros = []
for x in range(len(H_mod)):
    if H_mod[x] < 0.000001:
        ceros.append(x)

modulo = np.abs(H)

fase = np.angle(H)

plt.figure()
plt.plot(omega,abs(H))
plt.axvline(x=np.pi)
plt.show

plt.figure()
plt.plot(np.cos(omega), np.sin(omega))
for x in ceros:
    plt.plot(np.cos(x*np.pi/4000), np.sin(x*np.pi/4000), marker = 'o')
plt.show

plt.figure()
plt.plot(omega, fase)
plt.axvline(x=np.pi)
plt.show