import numpy as np
import sympy as sp

# Definir variables
x, y = sp.symbols('x y')

# Definir función vectorial F(x, y)
F = sp.Matrix([
    x**3 - y,   # Primera ecuación
    x**2 + y**2 - 1  # Segunda ecuación
])

# Calcular Jacobiano
J = F.jacobian([x, y])

# Función de Newton-Raphson para sistemas de ecuaciones no lineales
def newton(f, x0, jaco, tol=1e-6, max_iter=100):
    x_k = np.array(x0, dtype=float)  # Convertir x0 a array numérico
    vars_ = list(f.free_symbols)  # Obtener las variables (x, y en este caso)

    for i in range(max_iter):
        # Sustituir valores numéricos en F y J
        F_eval = np.array(f.subs({vars_[0]: x_k[0], vars_[1]: x_k[1]}), dtype=float).flatten()
        J_eval = np.array(jaco.subs({vars_[0]: x_k[0], vars_[1]: x_k[1]}), dtype=float)

        # Resolver el sistema lineal J * Δx = -F
        delta_x = np.linalg.solve(J_eval, -F_eval)

        # Actualizar x_k
        x_k = x_k + delta_x

        # Condición de convergencia
        if np.linalg.norm(delta_x) < tol:
            return x_k  # Retorna la raíz encontrada

    raise ValueError("El método de Newton no convergió en las iteraciones dadas")

# Punto inicial
x0 = [0.5, 0.5]  

# Ejecutar Newton
sol = newton(F, x0, J)

print("Solución encontrada:", sol)