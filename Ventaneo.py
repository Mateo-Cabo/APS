#%% módulos y funciones a importar

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy

#%% Datos de la simulación

fs = 1000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras
## Esto normaliza la resolucion espectral

# Datos del ADC
B =  8# bits
Vf = 2# rango simétrico de +/- Vf Volts
vref = np.sqrt(2)
q = (vref)/(2**(B-1))# paso de cuantización de q Volts
particiones = 2**B

# datos del ruido (potencia de la señal normalizada, es decir 1 W)
pot_ruido_cuant = (q**2)/12# Watts 
kn = 1 # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn # 

ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral


#%% Experimento: 

# Señales

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
    
    # print("SEÑAL SENOIDAL " + str(f0) + "Hz")
    # print("Número de muestras totales: " + str(N))
    # print("Número de muestras por ciclo: " + str(nn))
    # print("Frecuencia de muestreo: " + str(fs) + "Hz")
    # print("Frecuencia de la función a analizar: " + str(f0) + "Hz")
    # print("Voltaje máximo: " + str(vmax) + "V")
    # print("Voltaje acoplado en corriente continua: " + str(dc) + "V")
    # print("Fase de la función a analizar: " + str(ph) + "\n")
    
    return tt, xx

ff1 = 250
ff2 = 250
ff3 = 249.5
vmax = vref
dc = 0
ph = 0

# N: Número de muestras totales
# nn: Número de muestras por ciclo
# fs: Frecuencia de muestreo
# f0: Frecuencia de la función a analizar
# vmax: Voltaje máximo
# dc: Voltaje acoplado en corriente continua
# ph: Fase de la función a analizar

tt, xx = mi_funcion_sen( vmax, dc, ff1, ph, N, fs)
tt2, xx2 = mi_funcion_sen( vmax, dc, ff2, ph, N, fs)
tt3, xx3 = mi_funcion_sen( vmax, dc, ff3, ph, N, fs)

#%%Ventaneo

ventana = scipy.signal.windows.bohman(N)

zz = xx * ventana

#%% Normalizo
xn = xx/np.std(xx)
zzw = zz/ np.std(zz)

nn =  np.random.normal(0, np.sqrt(pot_ruido_analog), N) # señal de ruido de analógico

analog_sig = xx # señal analógica sin ruido
analog_sig2 = xx2 # señal analógica sin ruido
analog_sig3 = xx3 # señal analógica sin ruido

sr = analog_sig + nn # señal analógica de entrada al ADC (con ruido analógico)
srq = np.round(sr/q)*q # señal cuantizada

nq =  srq - sr # señal de ruido de cuantización/ruido digital



#%% Visualización de resultados

# cierro ventanas anteriores
plt.close('all')

###########
# Espectro
###########

plt.figure(2)
ft_SR = 1/N*np.fft.fft( sr)
ft_Srq = 1/N*np.fft.fft( srq)
ft_As = 1/N*np.fft.fft( zzw)
ft_As2 = 1/N*np.fft.fft( xn)

ft_Nq = 1/N*np.fft.fft( nq)
ft_Nn = 1/N*np.fft.fft( nn)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='blue', label='$ s $ (sig.)', marker = 'x' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As2[bfrec])**2), color='orange', label='$ s $ (sig.)' , marker = 'o')

plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()



