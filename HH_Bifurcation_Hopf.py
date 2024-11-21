import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from HH_Modelo import *

# Parametros obtenidos de minimizacion
C_m = 1.25053459
g_Na = 119.99924224
g_K = 36.02998743
g_L = 0.70246232 
V_Na = 49.99638308
V_K = -76.95790628 
V_L = -54.37535052 

#C_m = 1.0
#g_Na = 120.0
#g_K = 36.0
#g_L = 0.3 
#V_Na = 50.0 
#V_K = -77.0 
#V_L = -54.4 


I_range = np.linspace(0, 200, 200)
t_max=500
dt=0.01

max_values = []
min_values = []

for I_ext in I_range:
    max_val, min_val = find_min_max(I_ext,C_m,g_Na,g_K,g_L,V_Na,V_K,V_L,t_max, dt)
    if max_val is not None and min_val is not None:
        max_values.append(max_val)
        min_values.append(min_val)
    else:
        max_values.append(np.nan)
        min_values.append(np.nan)

plt.figure(figsize=(10, 6))
plt.plot(I_range, max_values,'.',label='Valor Máx Oscilación',markeredgewidth=0.05)
plt.plot(I_range, min_values,'.',label='Valor Mín Oscilación',markeredgewidth=0.05,color='#828282')
plt.axvline(x = 11.691691691691691, color = 'black',linestyle= '--', label = 'Valor crítico de I')
plt.xlabel('Corriente Externa (I)')
plt.ylabel('Potencial de Membrana (mV)')
plt.legend()
plt.grid()
plt.show()
#11.691691691691691
#6.25625626