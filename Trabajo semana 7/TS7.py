# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 10:57:26 2025

@author: mateo
"""

from pytc2.sistemas_lineales import plot_plantilla
import pytc2.filtros_digitales

import pytc2

import numpy as np
from scipy import signal as sig

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

ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])

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
qrs_detections = mat_struct['qrs_detections'].flatten()


plt.figure()
plt.plot(ecg_one_lead)
plt.show()

N = len(ecg_one_lead)

ecg_signal = ecg_one_lead / np.std(ecg_one_lead)
heartbeat_normal = heartbeat_normal / np.std(heartbeat_normal)
heartbeat_ventricular = heartbeat_ventricular / np.std(heartbeat_ventricular)
qrs_pattern = qrs_pattern / np.std(qrs_pattern)

#%% Ubicar una onda

w_after = 350 #ms
w_before = 250 #ms

ecg_segment = ecg_signal[qrs_detections[0] - w_before : qrs_detections[0] + w_after]
t_segment = np.arange(0, len(ecg_segment))

plt.figure()
plt.plot(t_segment, ecg_segment, label = 'Latido típico')
plt.xlabel('Tiempo[ms]')
plt.ylabel('Amplitud')
plt.title('Zoom')
plt.legend()
plt.show

#%% Filtros - definicion

#Plantilla de diseño/analisis

nyq_frec= fs_ecg/2

aprox_name = 'butter'

fpass = np.array( [1.0, 35.0] ) 
ripple = 0.25 # dB #gpass
fstop = np.array( [0.1, 50.0] ) #gstop
attenuation = 20 # dB

frecs = [0, 0.1, 1, 35, 50, nyq_frec]
gain = [0, -attenuation, -ripple, -ripple, -attenuation, -np.inf]

gain = 10**(np.array(gain)/20)

sos_butter = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype = aprox_name, output='SOS', fs=fs_ecg)


plt.figure()

npoints = 1000

w_rad = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi #muestreo en puntos donde yo quiera. Ya no es equiespaciado. Es una plantilla logaritmica

w, hh = sig.sosfreqz(sos_butter, worN=w_rad) #w = frecuencia, hh = array de complejos

fase = np.angle(hh)

fase_des = np.unwrap(fase)

modulo = np.abs(hh)

dH = np.diff(hh)
dw = np.diff(w_rad)

retardo = dH/-dw #derivado de la fase

plt.plot(w/np.pi*nyq_frec, 20*np.log10(np.abs(hh)+1e-15), label='mi_sos')

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

ax = plt.gca()

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs_ecg)
plt.legend()
plt.show()

#%% Aplicacion del filtro

ecg_filt_butter = sig.sosfiltfilt(sos = sos_butter, x = ecg_signal, axis=0, padtype='odd', padlen=0)

ecg_filt_butter = ecg_filt_butter / np.std(ecg_filt_butter)

plt.figure()
plt.plot(ecg_signal)
plt.plot(ecg_filt_butter)
plt.show()

#%% Cauer
aprox_name = 'ellip'

sos_cauer = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype = aprox_name, output='SOS', fs=fs_ecg)

plt.figure()

npoints = 1000

w_rad = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi #muestreo en puntos donde yo quiera. Ya no es equiespaciado. Es una plantilla logaritmica

w, hh = sig.sosfreqz(sos_cauer, worN=w_rad) #w = frecuencia, hh = array de complejos

fase = np.angle(hh)

fase_des = np.unwrap(fase)

modulo = np.abs(hh)

dH = np.diff(hh)
dw = np.diff(w_rad)

retardo = dH/-dw #derivado de la fase

plt.plot(w/np.pi*nyq_frec, 20*np.log10(np.abs(hh)+1e-15), label='mi_sos')

plt.title('Plantilla de diseño - Cauer')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

ax = plt.gca()

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs_ecg)
plt.legend()
plt.show()

tramo = np.arange(100000-9000,200000-9000,1)

ecg_filt_cauer = sig.sosfiltfilt(sos = sos_cauer, x = ecg_signal, axis=0, padtype='odd', padlen=0)
ecg_filt_cauer = ecg_filt_cauer / np.std(ecg_filt_cauer)

plt.figure()
plt.plot(ecg_signal)
plt.plot(ecg_filt_cauer)
plt.show()

#%% FIR
cant_coef = 2907

frecs = [0, 0.1, 3, 35, 50, fs_ecg/2]
gain = [0, -attenuation, -ripple, -ripple, -attenuation, -np.inf]

gain = 10**(np.array(gain)/20)

h = sig.firwin2(cant_coef, frecs, gain, window = ('kaiser', 6), fs=fs_ecg)

plt.figure()

npoints = 1000

w_rad = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi #muestreo en puntos donde yo quiera. Ya no es equiespaciado. Es una plantilla logaritmica

w, hh = sig.freqz(h, worN=w_rad)

plt.plot(w/np.pi*nyq_frec, 20*np.log10(np.abs(hh)+1e-15), label='h')

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

ax = plt.gca()

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs_ecg)
plt.legend()
plt.show()

#ecg_filt_kaiser = sig.lfilter(h,1,ecg_signal[tramo])
#t = np.arange(len(ecg_signal[tramo])) / fs_ecg

ecg_filt_kaiser = sig.filtfilt(h, 1, x = ecg_signal.flatten(), axis=-1, padtype='odd', padlen=0)
ecg_filt_kaiser = ecg_filt_kaiser / np.std(ecg_filt_kaiser)

plt.figure()
plt.plot(ecg_signal)
plt.plot(ecg_filt_kaiser)
plt.show()

 #%% Otro FIR

# def plot_freq_resp_fir(this_num, this_desc):

#     wrad, hh = sig.freqz(this_num, 1.0)
#     ww = wrad / np.pi
    
#     plt.figure(1)

#     plt.plot(ww, 20 * np.log10(abs(hh)), label=this_desc)

#     plt.title('FIR diseñado por métodos directos - Taps:' + str(cant_coef) )
#     plt.xlabel('Frequencia normalizada')
#     plt.ylabel('Modulo [dB]')
#     plt.grid(which='both', axis='both')

#     axes_hdl = plt.gca()
#     axes_hdl.legend()
    
#     plt.figure(2)

#     phase = np.unwrap(np.angle(hh))

#     plt.plot(ww, phase, label=this_desc)

#     plt.title('FIR diseñado por métodos directos - Taps:' + str(cant_coef))
#     plt.xlabel('Frequencia normalizada')
#     plt.ylabel('Fase [rad]')
#     plt.grid(which='both', axis='both')

#     axes_hdl = plt.gca()
#     axes_hdl.legend()

#     # plt.figure(3)

#     # # ojo al escalar Omega y luego calcular la derivada.
#     # gd_win = sig.group_delay(wrad, phase)

#     # plt.plot(ww, gd_win, label=this_desc)

#     # plt.ylim((np.min(gd_win[2:-2])-1, np.max(gd_win[2:-2])+1))
#     # plt.title('FIR diseñado por métodos directos - Taps:' + str(cant_coef))
#     # plt.xlabel('Frequencia normalizada')
#     # plt.ylabel('Retardo [# muestras]')
#     # plt.grid(which='both', axis='both')

#     # axes_hdl = plt.gca()
#     # axes_hdl.legend()   

# # tamaño de la respuesta al impulso
# cant_coef = 207

# filter_type = 'bandpass'

# fpass = np.array( [1.0 / nyq_frec, 35.0 / nyq_frec] ) 
# ripple = 1 # dB #gpass
# fstop = np.array( [0.1 / nyq_frec, 50.0 / nyq_frec] ) #gstop
# attenuation = 40 # dB

# # construyo la plantilla de requerimientos
# frecs = [0, 0.1/nyq_frec, 1/nyq_frec, 35/nyq_frec, 50/nyq_frec, 1]
# gain = [-np.inf, -attenuation, -ripple, -ripple, -attenuation, -np.inf] #dB

# gain = 10**(np.array(gain)/20)

# num_firls = sig.firls(cant_coef, frecs, gain, fs=fs_ecg/nyq_frec)
# plot_freq_resp_fir(num_firls, filter_type + '-firls')    

# fir_ls = pytc2.filtros_digitales.fir_design_ls()

# plt.figure(1)    
# plot_plantilla(filter_type = filter_type , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs_ecg/nyq_frec)
# axes_hdl = plt.gca()
# axes_hdl.legend()

#%% 
# Parámetros comunes del filtro Pasa Banda
order = 277  # Aumentamos el orden para mejor selectividad
fs = 2.0
fpass1 = 3 / nyq_frec  # Frecuencia de paso inferior
fpass2 = 45 / nyq_frec  # Frecuencia de paso superior
fstop1 = 0.001 / nyq_frec  # Frecuencia de corte inferior
fstop2 = 50 / nyq_frec  # Frecuencia de corte superior
ripple = 0.5 # dB
attenuation = 20  # dB
lgrid = 20

# Definir las bandas de paso y corte
band_edges = [0, fstop1, fpass1, fpass2, fstop2, 1.0]
desired = [0, 0, 1, 1, 0, 0]

# Desviaciones permisibles en dB
d = np.array([-ripple, -attenuation])  # En dB
d = 10**(d/20)  # Convertir de dB a lineal
d = np.array([(1-d[0]), d[1]])

# Diseño del filtro con LS
b_ls = pytc2.filtros_digitales.fir_design_ls(order, band_edges, desired, grid_density=lgrid, fs=fs, filter_type='m')

# Comparar la respuesta en frecuencia de ambos filtros
w, h_ls = sig.freqz(b_ls, worN=8000)

# Graficar ambas respuestas
plt.figure(figsize=(10, 6))
plt.plot(w/np.pi, 20 * np.log10(np.abs(h_ls)), label='LS - Mínimos Cuadrados', linestyle='-')
plot_plantilla(filter_type = 'bandpass' , fpass = (fpass1, fpass2), ripple = ripple , fstop = (fstop1, fstop2), attenuation = attenuation, fs = fs)
plt.title('Comparación de Filtros FIR Pasa Banda - PM vs LS')
plt.xlabel('Frecuencia Normalizada (×π rad/sample)')
plt.ylabel('Amplitud [dB]')
plt.legend()
plt.grid()
plt.show()

#ecg_filt_ls = sig.lfilter(b_ls,1,ecg_signal)
#t = np.arange(len(tramo)) / fs_ecg

ecg_filt_ls = sig.filtfilt(b_ls, [1], ecg_one_lead.flatten(), axis=-1, padtype='odd', padlen=1000)
ecg_filt_ls = ecg_filt_ls / np.std(ecg_filt_ls)

plt.figure()
plt.plot(ecg_signal)
plt.plot(ecg_filt_ls)
plt.show()

 
###################################
#%% Regiones de interés con ruido #
###################################
 
cant_muestras = len(ecg_signal)

regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_signal[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_filt_butter[zoom_region], label='ECG filt butter', linewidth=2)
    plt.plot(zoom_region, ecg_filt_cauer[zoom_region], label='ECG filt cauer', linewidth=2)
    plt.plot(zoom_region, ecg_filt_kaiser[zoom_region], label='ECG filt kaiser', linewidth=2)
    plt.plot(zoom_region, ecg_filt_ls[zoom_region], label='ECG filt ls', linewidth=2)

    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
    #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
 
###################################
#%% Regiones de interés sin ruido #
###################################
 
regs_interes = (
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_signal[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_filt_butter[zoom_region], label='butter', linewidth=2)
    plt.plot(zoom_region, ecg_filt_cauer[zoom_region], label='cauer', linewidth=2)
    plt.plot(zoom_region, ecg_filt_kaiser[zoom_region], label='kaiser', linewidth=2)
    plt.plot(zoom_region, ecg_filt_ls[zoom_region], label='ls', linewidth=2)


    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
   # plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()