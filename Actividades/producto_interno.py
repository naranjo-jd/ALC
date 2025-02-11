import numpy as np
import sympy as sp

# Definimos la variable simbolica x
x = sp.symbols('x')

# Definimos el conjunto de funciones
functions = [
    sp.Lambda(x, 1),
    sp.Lambda(x, sp.cos(x)),
    sp.Lambda(x, sp.sin(x)),
    sp.Lambda(x, sp.cos(2 * x)),
    sp.Lambda(x, sp.sin(2 * x))
]

# Definimos el producto interno
def inner_product(f, g, a, b):
    return sp.integrate(f(x) * g(x), (x, a, b)).doit()

# Definimos una funcion que verifique si dos funciones son ortogonales
def are_orthogonal(f, g, a, b):
    return inner_product(f, g, a, b) == 0

# Definimos una funcion que comprueba si el conjunto de funciones es ortogonal dos a dos
def check_othogonality(F, ip, a, b):
    for i in range(len(F)):
        for j in range(i+1, len(F)):
            if ip(F[i], F[j], a, b) != 0:
                print(f"Las funciones f{i+1} y f{j+1} NO son ortogonales.\n")
                return
    print("El conjunto es ortogonal dos a dos")
    
check_othogonality(functions, inner_product, -sp.pi, sp.pi)

# Definimos una funcion que calcula el vector de coeficientes con la base ortogonal
def coef_fourier(f, F, ip, a, b):
    coefficients = []
    for i in F:
        coef = ip(f, i, a, b)/ip(i, i, a, b)
        coefficients.append(coef)
    return coefficients

cubica = sp.Lambda(x, x**3)
print(coef_fourier(cubica, functions, inner_product, -sp.pi, sp.pi))
