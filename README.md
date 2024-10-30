# TesisMMA
Repositorio de Código para la Tesis "Análisis Dinámico del Comportamiento de Señales Cerebrales Humanas"

"HH_Modelo" incluye un conjunto de funciones a ser utilizadas en los demás scripts.

"HH_chaos" resuelve el sistema HH para los valores vanilla y el valor de bifurcacion de corriente externa (I_ext).

"HH_chaos_fitted" resuelve el sistema HH para los valores obtenidos de optimizacion.

"HH_Bifurcation_Hopf" muestra diagrama de bifurcacion sobre valores obtenidos de optimizacion.

"HH_fitted_SDK_Human_S" resuelve el modelo HH e implementa el modelo de optimizacion.

"HH_Hopf_Complex_Plane" resuelve HH iterativamente para valores de I_ext y grafica los eigenvalores en el plano complejo para demostrar bifurcacion de Hopf.

"HH_Jacobian" calcula la jacobiana y sus eigenvalores y eigenvectores en equilibrio.

----------------------------------------------------------------------------------

Se busca que "HH_fitted_SDK_Human", "HH_Hopf_Complex_Plane", "HH_Jacobian" logren depender completamente de "HH_Modelo"