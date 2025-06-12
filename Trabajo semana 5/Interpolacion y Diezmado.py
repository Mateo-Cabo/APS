# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 20:27:20 2025

@author: mateo
"""
import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
#fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
#fs_audio, wav_data = sio.wavfile.read('silbido.wav')

plt.figure()
plt.plot(wav_data)

# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
# import sounddevice as sd
#sd.play(wav_data, fs_audio)

#%% Diezmado

#%% 1. Filtrado

#BW = 1302 Hz

cant_coef = 2777

M = 2 #Diezmado x 

# fs = fs_audio # Hz
# nyq_frec= fs/2
# fpass = np.array( [1.0, 35.0] ) #Hz
# ripple = 1.0 # dB #gpass
# fstop = np.array( [0.1, 50.0] ) #gstop
# attenuation = 40 # dB

# frecs = [0, 0.1, 1, 35, 50, fs/2]
# gain = [0, -attenuation, -ripple, -ripple, -attenuation, -np.inf]

# gain = 10**(np.array(gain)/20)

# h = sig.firwin2(cant_coef, frecs, gain, window = ('kaiser', 14), fs=fs)

# #Plantilla de diseño/analisis

# npoints = 1000

# w_rad = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
# w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi #muestreo en puntos donde yo quiera. Ya no es equiespaciado. Es una plantilla logaritmica

# w, hh = sig.freqz(h, worN=w_rad)

# plt.plot(w/np.pi*nyq_frec, 20*np.log10(np.abs(hh)+1e-15), label='h')

#%% 2. Diezmado

audio_diez = wav_data[::M]

audio_diez2 = wav_data[::2*M]

audio_diez3 = wav_data[::4*M]

#%% Espectros

plt.figure()
plt.figure(figsize=(20,10))
plt.title('Señal de audio')
plt.plot(wav_data)

N1 = len(wav_data)
N2 = len(wav_data)
N3 = len(wav_data)

nperseg1 = N1 // 6
nperseg2 = N2 // 6
nperseg3 = N3 // 6

audio_signal1 = audio_diez / np.std(audio_diez)
audio_signal2= audio_diez2 / np.std(audio_diez2)
audio_signal3 = audio_diez3 / np.std(audio_diez3)

fs_audio_welch1, audio_sig_welch1 = sig.welch(x = audio_signal1, fs = fs_audio, window='hann', nperseg=nperseg1, noverlap=nperseg1//2, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=0, average='median')
fs_audio_welch2, audio_sig_welch2 = sig.welch(x = audio_signal2, fs = fs_audio, window='hann', nperseg=nperseg2, noverlap=nperseg2//2, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=0, average='median')
fs_audio_welch3, audio_sig_welch3 = sig.welch(x = audio_signal3, fs = fs_audio, window='hann', nperseg=nperseg3, noverlap=nperseg3//2, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=0, average='median')

signal_db1 =  10 * np.log10(audio_sig_welch1)
signal_db2 =  10 * np.log10(audio_sig_welch2)
signal_db3 =  10 * np.log10(audio_sig_welch3)

#%% Estimacion del ancho de banda

pot_total1 = np.sum(audio_sig_welch1)
pot_total2 = np.sum(audio_sig_welch2)
pot_total3 = np.sum(audio_sig_welch3)

pot_acumulada1 = np.cumsum(audio_sig_welch1)
pot_acumulada2 = np.cumsum(audio_sig_welch2)
pot_acumulada3 = np.cumsum(audio_sig_welch3)
condicion_95_1 = 0.95 * pot_total1
condicion_95_2 = 0.95 * pot_total2
condicion_95_3 = 0.95 * pot_total3

umbral_95 = np.where(pot_acumulada >= condicion_95)[0][0]

bw_95_au = fs_audio_welch[umbral_95]

plt.figure(4)
plt.figure(figsize=(20,10))
plt.title('Análsis espectral de una señal de audio')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia [dB]')

plt.plot(fs_audio_welch, signal_db, label = 'Espectometría de "La Cucaracha"')
plt.axvline(bw_95_au, color='r', linestyle='--', label='Umbral 95%')
plt.legend()

plt.show()
