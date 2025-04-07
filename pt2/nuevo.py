import sympy as sp
import numpy as np

def gradient(f, vars):
    return [sp.diff(f, var) for var in vars]

def jacobian(F, vars):
    return sp.Matrix([gradient(f, vars) for f in F])

def gradient_descent(f, vars, x0, alpha=0.1, tol=1e-6, max_iter=100):
    grad_f = gradient(f, vars)  # Calcula el gradiente simbólico
    grad_f_fun = sp.lambdify(vars, grad_f, 'numpy')  # Convierte a función numérica

    x = np.array(x0, dtype=float)  # Inicializa x0 como array numérico

    for _ in range(max_iter):
        grad_val = np.array(grad_f_fun(*x), dtype=float)  # Evalúa el gradiente en x
        if np.linalg.norm(grad_val) < tol:  # Criterio de convergencia
            break
        x -= alpha * grad_val  # Actualiza x en dirección contraria al gradiente

    return x

def gauss_newton(F, vars, x0, tol=1e-6, max_iter=100):
    J_func = sp.lambdify(vars, F.jacobian(vars), 'numpy')  # Jacobiano
    F_func = sp.lambdify(vars, F, 'numpy')  # Función de error

    x = np.array(x0, dtype=float)

    for _ in range(max_iter):
        J_val = np.array(J_func(*x), dtype=float)
        F_val = np.array(F_func(*x), dtype=float).squeeze()

        if np.linalg.norm(F_val) < tol:
            break

        delta_x = np.linalg.lstsq(J_val.T @ J_val, -J_val.T @ F_val, rcond=None)[0].flatten()
        x += delta_x  # Actualiza x

    return x

def newton_multivariate(F, vars, x0, tol=1e-6, max_iter=100):
    J_func = sp.lambdify(vars, F.jacobian(vars), 'numpy')  # Jacobiano simbólico -> numérico
    F_func = sp.lambdify(vars, F, 'numpy')  # Función simbólica -> numérica

    x = np.array(x0, dtype=float)

    for _ in range(max_iter):
        J_val = np.array(J_func(*x), dtype=float)
        F_val = np.array(F_func(*x), dtype=float).squeeze()

        if np.linalg.norm(F_val) < tol:
            break

        try:
            delta_x = np.linalg.solve(J_val, -F_val)  # Usamos la matriz Jacobiana completa
        except np.linalg.LinAlgError:
            raise ValueError("Jacobian is singular or ill-conditioned")

        x += delta_x  # Actualiza x

    return x