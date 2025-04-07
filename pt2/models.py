import numpy as np
import sympy as sp

def generate_matrices(points, degree=1):
    """
    Generates the design matrix A and vector b for least squares regression.

    Parameters:
        points (array-like): Nx2 array or list of (x, y) points.
        degree (int): Degree of the polynomial fit (default is 1 for linear regression).

    Returns:
        A (numpy array): The design matrix.
        b (numpy array): The target vector.
    """
    # Convert to NumPy array if not already
    points = np.asarray(points)  # Ensures input works for lists and np arrays

    # Extract x and y values
    x_vals = points[:, 0]  # First column (x-values)
    b = points[:, 1]       # Second column (y-values)

    # Construct the design matrix A (Vandermonde matrix for polynomial fitting)
    A = np.vander(x_vals, N=degree+1, increasing=True)

    return A, b

def least_squares(A, b):
    """
    Solves the least squares problem Ax = b using the normal equations.
    
    Parameters:
        A (numpy.ndarray): The input matrix.
        b (numpy.ndarray): The right-hand side vector.
    
    Returns:
        numpy.ndarray: The least squares solution.
    """
    return np.linalg.solve(A.T @ A, A.T @ b)

def gradient_descent(A, b, lr=0.01, tol=1e-6, max_iter=1000):
    """
    Solves the least squares problem Ax = b using gradient descent.
    
    Parameters:
        A (numpy.ndarray): The input matrix.
        b (numpy.ndarray): The right-hand side vector.
        lr (float, optional): Learning rate (default is 0.01).
        tol (float, optional): Convergence tolerance (default is 1e-6).
        max_iter (int, optional): Maximum number of iterations (default is 1000).
    
    Returns:
        numpy.ndarray: The approximate solution.
    """
    
    grad = 2*np.dot(A.T, E)
    x = np.linalg.solve(A.T @ A, A.T @ b)
    grad = gradient(x)
    for i in range(max_iter):

        gradient = A.T @ (A @ x - b)
        grad_norm = np.linalg.norm(gradient)
        
        if grad_norm < tol:
            break

        x -= lr * gradient

    return x

def gradient(f, vars):
    return [sp.diff(f, var) for var in vars]

def jacobian(F, vars):
    return sp.Matrix([gradient(f, vars) for f in F])

def gradient_descent_multi(f, vars, x0, alpha=0.1, tol=1e-6, max_iter=100):
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