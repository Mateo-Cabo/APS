# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 16:29:50 2025

@author: mateo
"""

import scipy
import numpy as np
from scipy import signal as sig
from scipy import interpolate

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

#%%

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################x

#para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead'].flatten()

N = len(ecg_one_lead)

# plt.figure(1)
# plt.plot(ecg_one_lead[5000:12000])
# plt.show()

##################
## ECG sin ruido
##################

#ecg_one_lead = np.load('ecg_sin_ruido.npy')

qrs1 = mat_struct['qrs_pattern1']
qrs_det = mat_struct['qrs_detections']

ecg_seg = ecg_one_lead

t_seg = np.arange(0, len(ecg_seg))/fs_ecg

#%% Filtro no lineal - filtro de mediana

from scipy.signal import medfilt

#venetanas

win1_samples = 200
win2_samples = 1200

#Paridad. Si es par le sumo 1
if win1_samples % 2 == 0:
    win1_samples += 1
if win2_samples % 2 == 0:
    win2_samples += qrs1
    
#Primer filtro mediana
ecg_med1 = medfilt(ecg_seg, kernel_size = win1_samples)

#Segundo filtro de mediana
ecg_med2 = medfilt(ecg_med1, kernel_size = win1_samples)

#%% Cubic spline

t_prom = []
val_prom = []

for idx in POI:
    window = ecg_one_lead[idx : idx + 20]
    prom = np.mean(window)
    t_prom.append(idx / fs_ecg)
    val_prom.append(prom)

ecg_3spline = scipy.interpolate.CubicSpline(t_seg, ecg_seg)

#%% Plot

plt.figure()
plt.plot(t_seg, ecg_seg, label = 'Señal original')
plt.plot(t_seg, ecg_med2, label = 'Linea de base')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('ECG filtrado con dos etapas de mediana')
plt.legend()
plt.show()

plt.figure()
plt.plot(t_seg, ecg_seg, label = 'Señal original')
plt.plot(t_seg, ecg_3spline(t_seg), label = 'Linea de base')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('ECG filtrado con Cubic Spline')
plt.legend()
plt.show()