# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 18:59:35 2025

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

# Cargar el archivo CSV como un array de NumPy
#fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
fs_audio, wav_data = sio.wavfile.read('silbido.wav')

# plt.figure(1)
# plt.plot(wav_data)

N = len(wav_data)

nperseg = N // 6

audio_signal = wav_data / np.std(wav_data)

fs_audio_welch, audio_sig_welch = sig.welch(x = audio_signal, fs = fs_audio, window='hann', nperseg=nperseg, noverlap=nperseg//2, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=0, average='median')

signal_db =  10 * np.log10(audio_sig_welch)

signal_db -= max(signal_db)

plt.figure(2)
plt.plot(fs_audio_welch, signal_db)
plt.show()

#signal_db -= max(signal_db)

f, t_stft, Zxx = stft(audio_signal, fs=fs_audio, nperseg=256)

#%% Estimacion del ancho de banda

# pot_total = np.sum(wav_data)

# pot_acumulada = np.cumsum(wav_data)
# condicion_95 = 0.95 * pot_total

# umbral_95 = np.where(pot_acumulada >= condicion_95)[0][0]

# bw_95_ecg = fs_audio_welch[umbral_95]

# plt.figure(4)
# plt.figure(figsize=(20,10))
# plt.title('Analisis espectral señal ECG')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Potencia [dB]')

# plt.plot(fs_audio_welch, signal_db, label = 'Espectrometria ecg')
# plt.axvline(bw_95_ecg, color='r', linestyle='--', label = 'Umbral 95%')
# plt.legend()
# plt.show()

t = np.arange(len(audio_signal)) / fs_audio

# STFT
f, t_stft, Zxx = stft(audio_signal, fs=fs_audio, nperseg=40)

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
ax2.semilogy(fs_audio_welch, audio_sig_welch)
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

import pywt

# Escalas y CWT
scales = np.logspace(0, np.log10(150), num=100)  # 1 a 100 en logscale, pero igual serán convertidas a Hz

# wavelet = pywt.ContinuousWavelet('cmor1.5-1.0') 
# wavelet = pywt.ContinuousWavelet('mexh') #mexican hat
wavelet = pywt.ContinuousWavelet('gaus3') #gaussiana 3er orden, 3 momentos no nulos: cualquier cosa que sea suave de tu señal, hasta un polinomio de orden 3, no le va a dar bola, solo a las partes más abruptas

f_c = pywt.central_frequency(wavelet)  # devuelve frecuencia normalizada
Δt = 1.0 / fs_audio
frequencies = f_c / (scales * Δt)

coefficients, frec = pywt.cwt(wav_data[0:8000], scales, wavelet, sampling_period=Δt)

# Crear figura y ejes
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# Señal
ax1.plot(t, wav_data)
ax1.set_title("Señal")
ax1.set_ylabel("Amplitud")
ax1.set_xlim(t[0], t[-1])
plt
pcm = ax2.imshow(np.abs(coefficients),
           extent=[t[0], t[-1], scales[-1], scales[0]],  # nota el orden invertido para eje Y
           cmap='viridis', aspect='auto')
ax2.set_title("CWT con wavelet basada en $B_3(x)$")
ax2.set_xlabel("Tiempo")
ax2.set_ylabel("Escala")
cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Dejar espacio para colorbar a la derecha
plt.show()