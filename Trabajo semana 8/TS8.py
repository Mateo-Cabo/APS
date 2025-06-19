# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 17:07:41 2025

@author: mateo
"""

from pytc2.sistemas_lineales import plot_plantilla
import pytc2.filtros_digitales

import pytc2

import numpy as np
from scipy import signal as sig
from scipy import interpolate

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

#%% Señal ECG

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

# Variables
qrs_pattern = mat_struct['qrs_pattern1'].flatten()
heartbeat_normal = mat_struct['heartbeat_pattern1'].flatten()
heartbeat_ventricular = mat_struct['heartbeat_pattern2'].flatten()
qrs_det = mat_struct['qrs_detections'].flatten()


plt.figure(3)
plt.plot(ecg_one_lead)
plt.show()

N = len(ecg_one_lead)

ecg_one_lead = ecg_one_lead / np.std(ecg_one_lead)
ecg_signal = ecg_one_lead / np.std(ecg_one_lead)
heartbeat_normal = heartbeat_normal / np.std(heartbeat_normal)
heartbeat_ventricular = heartbeat_ventricular / np.std(heartbeat_ventricular)
qrs_pattern = qrs_pattern / np.std(qrs_pattern)

#%% Filtro no lineal - filtro de mediana

from scipy.signal import medfilt

#venetanas

win1_samples = 200
win2_samples = 600

#Paridad. Si es par le sumo 1
if win1_samples % 2 == 0:
    win1_samples += 1
if win2_samples % 2 == 0:
    win2_samples += qrs_pattern
    
#Primer filtro mediana
ecg_med1 = medfilt(ecg_signal, kernel_size = win1_samples)

#Segundo filtro de mediana
ecg_med2 = medfilt(ecg_med1, kernel_size = win1_samples)

ecg_filt_med = ecg_signal - ecg_med2

#%% Spline cubico

N_ecg = len(ecg_one_lead)
t_ecg = np.arange(N_ecg) / fs_ecg  # eje de tiempo en segundos


def estimate_baseline_spline(ecg_sig, qrs_det, fs, n0_ms, window_ms):
    """
    Estima la línea de base de una señal ECG usando spline cúbica,
    a partir de puntos en el segmento PQ (aprox. n0_ms antes del QRS).
    El valor se promedia en una ventana de window_ms alrededor del punto.
    
    Parámetros:
    - ecg_signal: señal ECG 1D
    - qrs_positions: índices donde se detectan los QRS (enteros)
    - fs: frecuencia de muestreo (Hz)
    - n0_ms: tiempo en ms para retroceder desde QRS (segmento PQ)
    - window_ms: ventana de promedio en ms
    
    Retorna:
    - baseline: línea de base interpolada (vector del mismo largo que ecg_signal)
    - baseline_times: tiempos de los puntos fiduciales usados para la spline
    - baseline_values: valores promedio en esos puntos
    """
    
    n0 = int(n0_ms * fs / 1000)          # conversión ms a muestras
    window = int(window_ms * fs / 1000)  # ventana en muestras
    
    t_ecg = np.arange(len(ecg_sig)) / fs
    baseline_times = []
    baseline_values = []

    for qrs in qrs_det:
        idx = qrs - n0
        if idx - window//2 < 0 or idx + window//2 >= len(ecg_sig):
            continue  # evitamos salirnos del vector

        window_data = ecg_signal[idx - window//2 : idx + window//2]
        baseline_times.append(t_ecg[idx])
        baseline_values.append(np.mean(window_data))
        
    spline_func = interpolate.CubicSpline(baseline_times, baseline_values)
    baseline = spline_func(t_ecg)
    return baseline, baseline_times, baseline_values

# Estimamos línea de base usando spline cúbica
baseline_full, base_times, base_vals = estimate_baseline_spline(ecg_signal, qrs_det, fs=fs_ecg, n0_ms=100, window_ms=20)
ecg_clean_full = ecg_signal - baseline_full

#%% Filtro adaptado

# Centrar señal y patrón para eliminar offset
ecg_centered = ecg_signal - np.mean(ecg_signal)
pattern_centered = qrs_pattern[::-1] - np.mean(qrs_pattern)

#  Filtrado por convolución (matched filter)
filtered_signal = np.convolve(ecg_centered, pattern_centered, mode='same')

#  Detección de picos en señal filtrada 
height_threshold = np.max(filtered_signal) * 0.4
distance_threshold = int(fs_ecg * 0.3)
peaks, _ = sig.find_peaks(filtered_signal, height=height_threshold, distance=distance_threshold)

# Comparación detección con referencia
tolerance = int(0.05 * fs_ecg)
TP = 0
FP = 0
detected = np.zeros(len(qrs_det), dtype=bool)

for p in peaks:
    match = False
    for i, qrs in enumerate(qrs_det):
        if not detected[i] and abs(p - qrs) <= tolerance:
            TP += 1
            detected[i] = True
            match = True
            break
    if not match:
        FP += 1

FN = len(qrs_det) - TP
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

#  Resultados detección 
plt.figure(figsize=(12, 4))
plt.plot(ecg_signal, label='ECG original')
plt.plot(filtered_signal / np.max(filtered_signal), label='Filtro adaptado (normalizado)')
plt.plot(peaks, ecg_signal[peaks], 'rx', label='Detecciones')
plt.plot(qrs_det, ecg_signal[qrs_det], 'go', label='QRS referencia')
plt.legend()
plt.title("Detección de latidos con filtro adaptado")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()