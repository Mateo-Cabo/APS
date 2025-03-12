# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def mi_funcion_sen (vmax, dc, f0, ph, N, fs):
    ts=1/fs
    df=fs/N
    nn = fs/f0
    
    tt = np.linspace(0, (N-1)*ts, N).flatten()

    arg = (2*(np.pi)*f0*tt)

    xx = dc + vmax*np.sin(arg + ph)
    # En caso de que el ph estuviese en grados
    # phr = (ph*np.pi)/180 
    
    # xx = signal.square(arg)
    
    print("SEÑAL SENOIDAL " + str(f0) + "Hz")
    print("Número de muestras totales: " + str(N))
    print("Número de muestras por ciclo: " + str(nn))
    print("Frecuencia de muestreo: " + str(fs) + "Hz")
    print("Frecuencia de la función a analizar: " + str(f0) + "Hz")
    print("Voltaje máximo: " + str(vmax) + "V")
    print("Voltaje acoplado en corriente continua: " + str(dc) + "V")
    print("Fase de la función a analizar: " + str(ph) + "\n")
    
    return tt, xx

def mi_funcion_sqr (vmax, dc, f0, ph, N, fs):
    ts=1/fs
    df=fs/N
    nn = fs/f0
    
    tt = np.linspace(0, (N-1)*ts, N).flatten()

    arg = (2*(np.pi)*f0*tt)
    
    xx = signal.square(arg)
    
    print("SEÑAL CUADRADA " + str(f0) + "Hz")
    print("Número de muestras totales: " + str(N))
    print("Número de muestras por ciclo: " + str(nn))
    print("Frecuencia de muestreo: " + str(fs) + "Hz")
    print("Frecuencia de la función a analizar: " + str(f0) + "Hz")
    print("Voltaje máximo: " + str(vmax) + "V")
    print("Voltaje acoplado en corriente continua: " + str(dc) + "V")
    print("Fase de la función a analizar: " + str(ph) + "\n")
    
    return tt, xx

N = 1000
fs = 1000
ff1 = 1
ff2 = 500
ff3 = 999
ff4 = 1001
ff5 = 2001
ff6 = 1
vmax = 1
dc = 0
ph = 0

# N: Número de muestras totales
# nn: Número de muestras por ciclo
# fs: Frecuencia de muestreo
# f0: Frecuencia de la función a analizar
# vmax: Voltaje máximo
# dc: Voltaje acoplado en corriente continua
# ph: Fase de la función a analizar

tt1, xx1 = mi_funcion_sen( vmax, dc, ff1, ph, N, fs)
tt2, xx2 = mi_funcion_sen( vmax, dc, ff2, ph, N, fs)
tt3, xx3 = mi_funcion_sen( vmax, dc, ff3, ph, N, fs)
tt4, xx4 = mi_funcion_sen( vmax, dc, ff4, ph, N, fs)
tt5, xx5 = mi_funcion_sen( vmax, dc, ff5, ph, N, fs)
tt6, xx6 = mi_funcion_sqr( vmax, dc, ff6, ph, N, fs)

#%%Presentación de gráficos

def funcion_plot (tt, xx, n, num):
    plt.figure(num)
    line_hdls = plt.plot(tt, xx)
    if (num==6):
        plt.title('Señal cuadrada ' + str(n) + 'Hz')
    else:
        plt.title('Señal senoidal ' + str(n) + 'Hz')       
    plt.xlabel('Tiempo [segundos]')
    plt.ylabel('Amplitud [V]')
    plt.show()
    
    return

funcion_plot(tt1, xx1, ff1, 1)
funcion_plot(tt2, xx2, ff2, 2)
funcion_plot(tt3, xx3, ff3, 3)
funcion_plot(tt4, xx4, ff4, 4)
funcion_plot(tt5, xx5, ff5, 5)
funcion_plot(tt6, xx6, ff6, 6)