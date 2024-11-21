import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Ecuaciones del Modelo HH
def alpha_m(V):     
    return  0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

def beta_m(V): 
    return 4 * np.exp(-(V + 65) / 18)

def alpha_h(V): 
    return 0.07 * np.exp(-(V + 65) / 20)

def beta_h(V): 
    return 1 / (1 + np.exp(-(V + 35) / 10))

def alpha_n(V): 
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

def beta_n(V): 
    return 0.125 * np.exp(-(V + 65) / 80)

def hh_model(t, y, I_ext,C_m,g_Na,g_K,g_L,V_Na,V_K,V_L):
    V, m, h, n = y
    dVdt = (I_ext - g_Na * m**3 * h * (V - V_Na) - g_K * n**4 * (V - V_K) - g_L * (V - V_L)) / C_m
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    return [dVdt, dmdt, dhdt, dndt]

def hh_model_wrapper(y,t,I_ext,C_m,g_Na,g_K,g_L,V_Na,V_K,V_L):
    V, m, h, n = y
    dVdt = (I_ext - g_Na * m**3 * h * (V - V_Na) - g_K * n**4 * (V - V_K) - g_L * (V - V_L)) / C_m
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    return [dVdt, dmdt, dhdt, dndt]

def simulate(I_ext, C_m,g_Na,g_K,g_L,V_Na,V_K,V_L,t_max=500, dt=0.01):
    t = np.arange(0, t_max, dt)
    y0 = [-65, 0.0529, 0.5961, 0.3177] 
    sol = solve_ivp(hh_model, [0, t_max], y0, args=(I_ext,C_m,g_Na,g_K,g_L,V_Na,V_K,V_L), t_eval=t, method='RK45')
    return sol.t, sol.y[0]

def find_min_max(I_ext,C_m,g_Na,g_K,g_L,V_Na,V_K,V_L, t_max=500, dt=0.01):
    t, V = simulate(I_ext,C_m,g_Na,g_K,g_L,V_Na,V_K,V_L,t_max, dt)
    peaks = [i for i in range(1, len(V)-1) if V[i-1] < V[i] > V[i+1]]
    troughs = [i for i in range(1, len(V)-1) if V[i-1] > V[i] < V[i+1]]
    if len(peaks) < 2 or len(troughs) < 2:
        return None, None
    max_values = V[peaks]
    min_values = V[troughs]
    return max_values[-1], min_values[-1]
