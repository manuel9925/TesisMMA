import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals, eig
from scipy.optimize import fsolve
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Parametros HH
C_m = 1.0  # capacitancia de membrana, en uF/cm^2
g_K = 36.0  # conductividad maxima de Na (mS/cm^2)
g_Na = 120.0  # conductividad maxima de K (mS/cm^2)
g_L = 0.3  # conductividad de fuga (mS/cm^2)
E_K = -77.0  # potencial de inversion de sodio (mV)
E_Na = 50.0  # potencial de inversion de potasio (mV)
E_L = -54.4 # potencial de inversion de fuga (mV)

# Definir modelo HH simplificado
def hodgkin_huxley(y, I_ext):
    V, n, m, h = y

    alpha_n = (0.01*(V+55)) / (1 - np.exp(-0.1*(V+55)))
    beta_n = 0.125 * np.exp(-0.0125*(V+65))
    
    alpha_m = (0.1*(V+40)) / (1 - np.exp(-0.1*(V+40)))
    beta_m = 4.0 * np.exp(-0.0556*(V+65))
    
    alpha_h = 0.07 * np.exp(-0.05*(V+65))
    beta_h = 1.0 / (1 + np.exp(-0.1*(V+35)))

    I_K = g_K * n**4 * (V - E_K)
    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_L = g_L * (V - E_L)
    
    dVdt = (I_ext - I_K - I_Na - I_L) / C_m
    dndt = alpha_n * (1 - n) - beta_n * n
    dmdt = alpha_m * (1 - m) - beta_m * m
    dhdt = alpha_h * (1 - h) - beta_h * h
    
    return [dVdt, dndt, dmdt, dhdt]

# Definiendo matriz Jacobiana numericamente
def jacobian(V, n, m, h, I_ext):
    # Pequeña perturbacion para diferenciacion numerica
    epsilon = 1e-6
    J = np.zeros((4, 4))
    
    # Calculando derivadas parciales numericamente
    for i, y_i in enumerate([V, n, m, h]):
        perturbed = np.array([V, n, m, h])
        perturbed[i] += epsilon
        f_perturbed = hodgkin_huxley(perturbed, I_ext)
        f_original = hodgkin_huxley([V, n, m, h], I_ext)
        
        J[:, i] = (np.array(f_perturbed) - np.array(f_original)) / epsilon
        
    return J

# Funcion para traer puntos de equilibrio
def find_equilibrium(I_ext):
    
    initial_guess = [-65, 0.0529, 0.5961, 0.3177]

    # Solucionando sistema para encontrar equilibrio
    equilibrium = fsolve(lambda y: hodgkin_huxley(y, I_ext), initial_guess)
    return equilibrium

# Valores a explorar de I_ext
#I_ext_values = np.linspace(6.25625626, 6.25625626, 1)  
I_ext_values = np.linspace(0, 100, 100) 

# Normalizando para mapa de colores
norm = mcolors.Normalize(vmin=min(I_ext_values), vmax=max(I_ext_values))
cmap = cm.viridis  

# Graficando eigenvalores en plano complejo
plt.figure(figsize=(8, 8))

# Iterando sobre diferentes valores de eigenvalores y jacobianas
for I_ext in I_ext_values:
    # Encontrando equilibrio para valor actual de I_ext
    V_eq, n_eq, m_eq, h_eq = find_equilibrium(I_ext)
    
    # Calculando Jacobiana en equilibrio
    J = jacobian(V_eq, n_eq, m_eq, h_eq, I_ext)
    
    # Calculando eigenvalores y eigenvectores de la Jacobiana
    eigenvals = eigvals(J)
    eigenvalores,eigenvectores= eig(J)
    
    # mapeando a color
    color = cmap(norm(I_ext))
    
    # Graficando
    plt.scatter(np.real(eigenvals), np.imag(eigenvals), color=color, alpha=0.7)

# Añadiendo barra de color
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='I_ext')

# Graficando
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel(r'$Re(\lambda)$')
plt.ylabel(r'$Im(\lambda)$')
plt.grid(True)
plt.show()
