import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from allensdk.core.cell_types_cache import CellTypesCache

# Creando instancia para llamda a API
ctc = CellTypesCache(manifest_file='cell_types/manifest.json')

# Esto salva el archivo NWB a 'cell_types/specimen_611598198/ephys.nwb'
cell_specimen_id = 611598198
data_set = ctc.get_ephys_data(cell_specimen_id)

# Numero de barrido
sweep_number = 45
sweep_data = data_set.get_sweep(sweep_number)

index_range = sweep_data["index_range"]
estimulo = sweep_data["stimulus"][0:index_range[1]+1] # en A
respuesta = sweep_data["response"][0:index_range[1]+1] # en A
estimulo *= 1e12 # convirtiendo a pA
respuesta *= 1e3 # convirtiendo a mV


window_init= np.where(estimulo==90)[0][0]
window_end= np.where(estimulo==90)[0][-1]

estimulo= estimulo[window_init:window_end]
respuesta= respuesta[window_init:window_end]
respuesta=respuesta[1300:2500]

sampling_rate = sweep_data["sampling_rate"] # in Hz


# Constantes para modelo 
C_m = 1.0  # capacitancia de membrana, en uF/cm^2
g_Na = 120.0  # conductividad maxima de Na (mS/cm^2)
g_K = 36.0  # conductividad maxima de K (mS/cm^2)
g_L = 0.3  # conductividad de fuga (mS/cm^2)
V_Na = 50.0  # potencial de inversion de sodio (mV)
V_K = -77.0  # potencial de inversion de potasio (mV)
V_L = -54.4  # potencial de inversion de fuga (mV)

# Array de Tiempo
t = np.linspace(0, 25, 1200) 
t= np.arange(0, len(respuesta)) * (1000.0 / sampling_rate)
I_ext = np.zeros_like(t)
I_ext[0:len(t)] = 17.0  

def alpha_m(V):
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

def beta_m(V):
    return 4.0 * np.exp(-(V + 65) / 18)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65) / 20)

def beta_h(V):
    return 1 / (1 + np.exp(-(V + 35) / 10))

def alpha_n(V):
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80)

def hh_model(y, t, params):
    V, m, h, n = y
    g_Na, g_K, g_L, E_Na, E_K, E_L, C_m = params  

    I_Na = g_Na * (m**3) * h * (V - E_Na)
    I_K = g_K * (n**4) * (V - E_K)
    I_L = g_L * (V - E_L)

    dVdt = (I_ext[int(t)] - (I_Na + I_K + I_L)) / C_m

    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n

    return [dVdt, dmdt, dhdt, dndt]

# Condiciones Iniciales
V0 = -65 
m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))
y0 = [V0, m0, h0, n0]  

# Known parameters for generating synthetic data
true_params = [g_Na, g_K, g_L, V_Na, V_K, V_L, C_m]

# Simulate using known parameters
synthetic_data = odeint(hh_model, y0, t, args=(true_params,))
V_synthetic = synthetic_data[:, 0] 

V_observed= respuesta
# Grafica la data sintetica
plt.plot(t, V_observed, label='respuesta')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Potencial de Membrana (mV)')
plt.legend()
plt.show()

# Funcion de pérdida: MSE entre voltaje simulado y observado
def loss_function(params, experimental_data, time_points):
     # Corre la simulacion con los parametros dados
    sim_data = odeint(hh_model, y0, time_points, args=(params,))
    
    # Extraer el voltaje simulado
    V_sim = sim_data[:, 0]
    
    # Calculando mean squared error (MSE)
    mse = np.sqrt(np.mean((V_sim - experimental_data)**2))
    return mse

# Intento inicial de los parametros
params_init = [120.0, 36.0, 0.3, 50.0, -77.0, -54.4, 1.0]


# Optimizando parametros utilizando funcion de pérdida
result = minimize(loss_function, params_init, args=(V_observed, t),tol=1e-6, method='L-BFGS-B',options={'maxiter': 10000, 'disp': True})

# Extrayendo parametros Optimizados
optimized_params = result.x

#result2 = minimize(loss_function, optimized_params, args=(V_observed, t),tol=1e-6, method='BFGS',options={'maxiter': 1000, 'disp': True})

#optimized_params2 = result2.x

#result3 = minimize(loss_function, optimized_params2, args=(V_observed, t),tol=1e-6, method='CG',options={'maxiter': 1000, 'disp': True})

#optimized_params3 = result3.x

print("Parametros Optimizados:", optimized_params)

# Simulando el modelo HH con parametros optimizados
fitted_data = odeint(hh_model, y0, t, args=(optimized_params,))
V_fitted = fitted_data[:, 0] 

# Graficando
plt.plot(t, V_observed, label='Data Observada', color='blue')
#plt.plot(t, V_synthetic, label='Data Sintetica', color='red', linestyle='dashed')
plt.plot(t, V_fitted, label='Modelo Ajustado', color='green', linestyle='dashed')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Potencial de Membrana (mV)')
plt.legend()
plt.show()


