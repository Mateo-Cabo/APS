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

omega = np.arange(0, 2*np.pi, np.pi/1000)

arg = omega  * 1j

z = np.exp(arg)

H = z**-4 + z**-3 + z**-2 + z**-1 + z**0

a = 250
b = 500
tolera = 1.01

tramo = b-a
while not(tramo<tolera):
    c = (a+b)//2
    fa = H.real[a]
    fb = H.real[b]
    fc = H.real[c]
    cambia = fa*fc
    if cambia < 0: 
        a = a
        b = c
    if cambia > 0:
        a = c
        b = b
    tramo = b-a

#500
#
#

plt.figure()
plt.plot(abs(H.real))
plt.plot(H.real)
plt.show

plt.figure()
plt.plot(np.cos(omega), np.sin(omega))
plt.plot(np.cos(((c-1))*np.pi/1000), np.sin(((c-1))*np.pi/1000), marker = 'o')
plt.show