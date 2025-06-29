# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 20:04:30 2025

@author: mateo
"""
import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

from scipy.signal import stft, welch

#%% Señal ECG

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################x

#para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = (mat_struct['ecg_lead']).ravel()
N = len(ecg_one_lead)


plt.figure(1)
plt.plot(ecg_one_lead[5000:12000])
plt.show()

##################
## ECG sin ruido
##################

# ecg_one_lead = np.load('ecg_sin_ruido.npy')

plt.figure(3)
plt.plot(ecg_one_lead)
plt.show()

N = len(ecg_one_lead)

nperseg = N // 8

audio_signal = ecg_one_lead / np.std(ecg_one_lead)

fs_ecg_welch, ecg_sig_welch = sig.welch(x = audio_signal, fs = fs_ecg, window='hann', nperseg=nperseg, noverlap=nperseg//2, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=0, average='median')

signal_db =  10 * np.log10(ecg_sig_welch)

#signal_db -= max(signal_db)

plt.figure(2)
plt.plot(fs_ecg_welch, signal_db)
plt.show()

t = np.arange(len(audio_signal)) / fs_ecg

# STFT
f, t_stft, Zxx = stft(audio_signal, fs=fs_ecg, nperseg=5)

# Crear figura y ejes
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)

# Señal
ax1.plot(t, audio_signal)
#ax1.set_title(f"Señal compuesta: {f1} Hz @ {center1}s y {f2} Hz @ {center2}s")
ax1.set_ylabel("Amplitud")
ax1.set_xlim(t[0], t[-1])

# Subplot 2: Welch
ax2.semilogy(fs_ecg_welch, ecg_sig_welch)
ax2.set_title("Estimación espectral por Welch")
ax2.set_ylabel("PSD [V²/Hz]")
ax2.set_xlabel("Frecuencia [Hz]")

# Espectrograma
pcm = ax3.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
ax3.set_title("STFT (Espectrograma)")
ax3.set_ylabel("Frecuencia [Hz]")
ax3.set_xlabel("Tiempo [s]")

# Colorbar en eje externo

cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Dejar espacio para colorbar a la derecha
plt.show()

#%%

# import pywt

# # Escalas y CWT
# scales = np.logspace(0, np.log10(150), num=100)  # 1 a 100 en logscale, pero igual serán convertidas a Hz

# # wavelet = pywt.ContinuousWavelet('cmor1.5-1.0') 
# # wavelet = pywt.ContinuousWavelet('mexh') #mexican hat
# wavelet = pywt.ContinuousWavelet('gaus3') #gaussiana 3er orden, 3 momentos no nulos: cualquier cosa que sea suave de tu señal, hasta un polinomio de orden 3, no le va a dar bola, solo a las partes más abruptas

# f_c = pywt.central_frequency(wavelet)  # devuelve frecuencia normalizada
# Δt = 1.0 / fs_audio
# frequencies = f_c / (scales * Δt)

# coefficients, frec = pywt.cwt(wav_data[0:8000], scales, wavelet, sampling_period=Δt)

# # Crear figura y ejes
# fig = plt.figure(figsize=(12, 6))
# ax1 = plt.subplot(2, 1, 1)
# ax2 = plt.subplot(2, 1, 2)

# # Señal
# ax1.plot(t, wav_data)
# ax1.set_title("Señal")
# ax1.set_ylabel("Amplitud")
# ax1.set_xlim(t[0], t[-1])
# plt
# pcm = ax2.imshow(np.abs(coefficients),
#            extent=[t[0], t[-1], scales[-1], scales[0]],  # nota el orden invertido para eje Y
#            cmap='viridis', aspect='auto')
# ax2.set_title("CWT con wavelet basada en $B_3(x)$")
# ax2.set_xlabel("Tiempo")
# ax2.set_ylabel("Escala")
# cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
# fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")

# plt.tight_layout(rect=[0, 0, 0.9, 1])  # Dejar espacio para colorbar a la derecha
# plt.show()