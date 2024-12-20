import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from HH_Modelo import *

# Constantes resultantes 
C_m = 1.0
g_Na = 120.0
g_K = 36.0
g_L = 0.3 
V_Na = 50.0 
V_K = -77.0 
V_L = -54.4 

# Condiciones Iniciales y vector de tiempo
y_0 = [-65, 0.0529, 0.5961, 0.3177]
t_max= 500
t = np.linspace(0, t_max, 1000000)
I = 6.25625626  # Corriente externa
#I =  10  # Corriente externa

# Soluciona el modelo
sol = solve_ivp(hh_model, [0, t_max], y_0, args=(I,C_m,g_Na,g_K,g_L,V_Na,V_K,V_L), t_eval=t, method='RK45')

# Graficando
plt.figure(figsize=(12, 8))
plt.plot(t, sol['y'][0], label='V (mV)')
plt.xlabel('Tiempo(ms)')
plt.ylabel('Potencial de Membrana (mV)')
plt.legend()
plt.show()

# Graficando diagramas de fase
plt.figure(figsize=(18, 6))

# V vs. m
plt.subplot(1, 3, 1)
plt.plot(sol['y'][0], sol['y'][1])
plt.xlabel('V (mV)')
plt.ylabel('m')

# V vs. h
plt.subplot(1, 3, 2)
plt.plot(sol['y'][0], sol['y'][2])
plt.xlabel('V (mV)')
plt.ylabel('h')

# V vs. n
plt.subplot(1, 3, 3)
plt.plot(sol['y'][0], sol['y'][3])
plt.xlabel('V (mV)')
plt.ylabel('n')

plt.tight_layout()
plt.show()


#fig = plt.figure()
#ax = plt.axes(projection='3d')

# Data for a three-dimensional line
#zline = sol['y'][3]
#xline = sol['y'][0]
#yline = sol['y'][1]
#ax.plot3D(xline, yline, zline, 'gray')
#fig.show()

# Rango parametrico para bifurcacion
t = np.linspace(0, 200, 10000)  
I_values = np.linspace(0, 10, 1000)

# Lista para data de bifurcacion
bifurcation_data = []

for I in I_values:
    sol = solve_ivp(hh_model, [0, t_max], y_0, args=(I,C_m,g_Na,g_K,g_L,V_Na,V_K,V_L), t_eval=t, method='RK45')
    V = sol['y'][0]
    
       # Removiendo data transitoria
    V = V[len(V)//2:]
    
    # Extrayendo maximo local
    local_maxima = V[(np.diff(np.sign(np.diff(V))) < 0).nonzero()[0] + 1]
    
    # Almacenando valor del parametro y maximo local
    for vmax in local_maxima:
        bifurcation_data.append((I, vmax))

# Convirtiendo a numpy array
bifurcation_data = np.array(bifurcation_data)

# Graficando diagrama de bifurcacion
plt.figure(figsize=(12, 8))
plt.plot(bifurcation_data[:, 0], bifurcation_data[:, 1], 'k.', markersize=2)
plt.axvline(x = 6.25625626, color = 'b',linestyle= '--', label = 'Valor crítico de I')
plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper left')
plt.xlabel('Corriente Externa (I)')
plt.ylabel('Potencial de Membrana (V)')
plt.grid(True)
plt.show()
