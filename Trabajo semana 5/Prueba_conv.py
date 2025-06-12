# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 20:13:16 2025

@author: mateo
"""

import numpy as np
from scipy import signal as sig

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

#%%Normalizacion

ecg_one_lead = ecg_one_lead/np.std(ecg_one_lead)
qrs1 = qrs1/np.std(qrs1)

#%% Correlacion

cor_ECG = np.correlate(ecg_one_lead, qrs1.flatten(), mode='same')
cor_ECG = cor_ECG/np.std(cor_ECG)

t_cor = np.arange(len(cor_ECG))/fs_ecg

#%% Deteccion de picos

threshold = 0.4
distance = 300

peaks, _ = sig.find_peaks(cor_ECG, height=threshold, distance=distance)
#%% Separar las ondas
ondas = np.zeros(shape=(len(peaks), 400))
j = 0
for x in peaks:
    i=0
    vector = np.zeros(400)
    while i<400:
        vector[i] = ecg_one_lead[x-100+i]
        i += 1
    ondas[j] = vector.transpose()
    j += 1

tipo1 = np.zeros(shape=(len(peaks), 400))
tipo2 = np.zeros(shape=(len(peaks), 400))
i = 0
k = 0
for x in range(len(ondas)):
    if np.max(ondas[x]) > 1.5:
        tipo1[i] = ondas[x]
        i += 1
    else:
        tipo2[k] = ondas[x]
        k += 1
#%% Plot

plt.figure()
plt.plot(t_cor, ecg_one_lead)
#plt.plot(t_cor[qrs_det], ecg_one_lead[qrs_det], 'x')
plt.plot(t_cor, cor_ECG)
plt.plot(t_cor[peaks], cor_ECG[peaks],'x')
plt.legend()
plt.show()

plt.figure()
for x in range(len(peaks)):
    plt.plot(np.arange(-100,300,1), ondas[x])
plt.show()


plt.figure()
for x in range(619):
    plt.plot(np.arange(-100,300,1), tipo2[x])
plt.show()


plt.figure()
for x in range(1259):
    plt.plot(np.arange(-100,300,1), tipo1[x])
plt.show()





