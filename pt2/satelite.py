import numpy as np
import sympy as sp
import models

# Variables
x, y, z, d = sp.symbols('x y z d')
vars = [x, y, z, d]

# Parámetros conocidos
c = 299792.458  # velocidad de la luz

# Coordenadas de los sensores y tiempos de llegada
A = np.array([
    [15600, 7540, 20140, 0.07074],   # Sensor 1: (x, y, z, t)
    [18760,2750,18610, 0.07220],
    [17610,14630,13480, 0.07690],
    [19170,610,18390, 0.07242]
])

# Construir funciones r_i
F = []
for Ai, Bi, Ci, ti in A:
    dist = sp.sqrt((x - Ai)**2 + (y - Bi)**2 + (z - Ci)**2)
    ri = dist - c * (ti - d)
    F.append(ri)

F = sp.Matrix(F)  # Sistema de ecuaciones

x0 = [0, 0, 6370, 0]

sol = models.newton_multivariate(F, vars, x0)
print(sol)