# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 21:24:05 2025

@author: mateo
"""

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

# plt.figure(1)
# plt.plot(ecg_one_lead[5000:12000])
# plt.show()

##################
## ECG sin ruido
##################

#ecg_one_lead = np.load('ecg_sin_ruido.npy')

qrs1 = mat_struct['qrs_pattern1']

#%%Normalizacion

ecg_one_lead = ecg_one_lead/np.std(ecg_one_lead)
qrs_norm = qrs1/np.std(qrs1)

#%% Correlacion

cor_ECG = np.correlate(ecg_one_lead, qrs_norm.flatten())
cor_ECG = cor_ECG/np.std(cor_ECG)

#%% Plot

plt.figure()
plt.plot(ecg_one_lead)
plt.plot(cor_ECG)
plt.legend()
plt.show()

#%% Filtro estadistico

