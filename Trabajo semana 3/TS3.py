#%% módulos y funciones a importar

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy

#%% Datos de la simulación

fs = 1000
fs2 = 18000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras

# Datos del ADC
B =  16# bits
Vf = 2# rango simétrico de +/- Vf Volts
vref = np.sqrt(2)/2
q = (Vf)/(2**(B-1))# paso de cuantización de q Volts
particiones = 2**B

# datos del ruido (potencia de la señal normalizada, es decir 1 W)
pot_ruido_cuant = (q**2)/12# Watts 
kn = 1/10 # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn # 

ts = 1/fs # tiempo de muestreo
ts2 = 1/fs2
df = fs/N # resolución espectral

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

def mi_funcion_sen2 (vmax, dc, f0, ph, N, fs):
    
    tt = np.linspace(0, (N-1)*ts2, N).flatten()

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

ff1 = 510 #fs/N
ff2 = 510
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
tt2, xx2 = mi_funcion_sen2( vmax, dc, ff2, ph, 18000, fs)
tt3, xx3 = mi_funcion_sen( vmax, dc, ff3, ph, N, fs)

#%% Normalizo
xn = xx/np.std(xx)
xn2 = xx2/np.std(xx2)

nn =  np.random.normal(0, np.sqrt(pot_ruido_analog), N) # señal de ruido de analógico

analog_sig = xn # señal analógica sin ruido
analog_sig2 = xn2 # señal analógica sin ruido
analog_sig3 = xx3 # señal analógica sin ruido

sr = analog_sig + nn # señal analógica de entrada al ADC (con ruido analógico)
srq = np.round(sr/q)*q # señal cuantizada

nq =  srq - sr # señal de ruido de cuantización/ruido digital



#%% Visualización de resultados

# cierro ventanas anteriores
plt.close('all')

##################
# Señal temporal
##################

plt.figure(1)

#plt.plot(tt, srq, lw=1, color='blue', markersize=5, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='ADC out (diezmada)')
plt.plot(tt, sr, lw=1, color='green', marker='o', ls='dotted', label='ADC in')
plt.plot(tt, analog_sig, lw=1, color='orange', label='$ s $ (analog)')
plt.plot(tt2, analog_sig2, lw=1, color='blue', ls='dotted', label='$ s $ (analog2)')

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()

###########
# Espectro
###########

plt.figure(2)
ft_SR = 1/N*np.fft.fft( sr)
ft_Srq = 1/N*np.fft.fft( srq)
ft_As = 1/N*np.fft.fft( xn)
ft_Nq = 1/N*np.fft.fft( nq)
ft_Nn = 1/N*np.fft.fft( nn)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='blue', label='$ s $ (sig.)', marker = 'x' )
##plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As2[bfrec])**2), color='orange', label='$ s $ (sig.)' , marker = 'o')
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As3[bfrec])**2), color='red', label='$ s $ (sig.)' )

plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

# plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

#############
# Histograma
#############

plt.figure(3)
bins = 10
plt.hist(nq.flatten()/(q), bins=bins)
plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

plt.xlabel('Pasos de cuantización (q) [V]')

# def funcion_plot (tt, xx, n, num, color):
#     plt.figure(num)
#     line_hdls = plt.plot(tt, xx, color = str(color))
#     if (num==6):
#         plt.title('Señal cuadrada ' + str(n) + 'Hz')
#     else:
#         plt.title('Señal senoidal ' + str(n) + 'Hz')       
#     plt.xlabel('Tiempo [segundos]')
#     plt.ylabel('Amplitud [V]')
    
#     return

# # funcion_plot(tt, xx, zzw, 4, str('blue'))
# # funcion_plot(tt2, xx2, ff2, 4, str('orange'))
# # funcion_plot(tt2, xx3, ff3, 4, str('red'))

# plt.show()


