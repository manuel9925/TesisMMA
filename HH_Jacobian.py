import numpy as np
from numpy.linalg import eig
from scipy.optimize import fsolve
from HH_Modelo import *

# Definiendo constantes
C_m = 1.25053459
g_Na = 119.99924224
g_K = 36.02998743
g_L = 0.70246232 
E_Na = 49.99638308 
E_K = -76.95790628 
E_L = -54.37535052

#C_m = 1.0
#g_Na = 120.0
#E_Na = 50.0
#g_K = 36.0
#E_K = -77.0
#g_L = 0.3
#E_L = -54.4

#I_ext = 6.25625626
I_ext= 11.691691691691691

# Definiendo modelo HH con inputs reducidos
def hodgkin_huxley(state):
    V, n, m, h = state
    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_L = g_L * (V - E_L)
    
    dV_dt = (I_ext - I_Na - I_K - I_L) / C_m
    dn_dt = alpha_n(V) * (1 - n) - beta_n(V) * n
    dm_dt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dh_dt = alpha_h(V) * (1 - h) - beta_h(V) * h
    
    return [dV_dt, dn_dt, dm_dt, dh_dt]

# Encontrando punto de equilibrio
def equations(state):
    return hodgkin_huxley(state)

initial_guess = [-65, 0.32, 0.05, 0.6]
equilibrium = fsolve(equations, initial_guess)

# Calculando jacobiana en punto de equilibrio
def jacobian(state):
    V, n, m, h = state
    J = np.zeros((4, 4))
    
    # Derivadas Parciales
    dI_Na_dV = g_Na * m**3 * h
    dI_Na_dm = 3 * g_Na * m**2 * h * (V - E_Na)
    dI_Na_dh = g_Na * m**3 * (V - E_Na)
    
    dI_K_dV = g_K * n**4
    dI_K_dn = 4 * g_K * n**3 * (V - E_K)
    
    dI_L_dV = g_L
    
    # Poblando matriz jacobiana
    J[0, 0] = -(dI_Na_dV + dI_K_dV + dI_L_dV) / C_m
    J[0, 1] = -dI_K_dn / C_m
    J[0, 2] = -dI_Na_dm / C_m
    J[0, 3] = -dI_Na_dh / C_m
    
    J[1, 0] = alpha_n(V) * (-1 / (1 - np.exp(-(V + 55) / 10)) - 0.01 * (V + 55) * np.exp(-(V + 55) / 10) / (1 - np.exp(-(V + 55) / 10))**2) - beta_n(V) * (-65 / 80) * np.exp(-(V + 65) / 80)
    J[1, 1] = -alpha_n(V) - beta_n(V)
    
    J[2, 0] = alpha_m(V) * (-1 / (1 - np.exp(-(V + 40) / 10)) - 0.1 * (V + 40) * np.exp(-(V + 40) / 10) / (1 - np.exp(-(V + 40) / 10))**2) - beta_m(V) * (-65 / 18) * np.exp(-(V + 65) / 18)
    J[2, 2] = -alpha_m(V) - beta_m(V)
    
    J[3, 0] = alpha_h(V) * (-1 / (1 - np.exp(-(V + 65) / 20)) - 0.07 * (V + 65) * np.exp(-(V + 65) / 20) / (1 - np.exp(-(V + 65) / 20))**2) - beta_h(V) * (-35 / 10) * np.exp(-(V + 35) / 10)
    J[3, 3] = -alpha_h(V) - beta_h(V)
    
    return J

jacobian_at_eq = jacobian(equilibrium)

# Calculando eigenvalores y eigenvectores
eigenvalues, eigenvectors = eig(jacobian_at_eq)

print(f"Punto de Equilibrio: {equilibrium}")
print(f"Matriz Jacobiana en equilibrio: \n{jacobian_at_eq}")
print(f"Eigenvalores: {eigenvalues}")
print(f"Eigenvectores: \n{eigenvectors}")
