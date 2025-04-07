import numpy as np
import sympy as sp

def gradient_descent(f_grad, x0, lr=0.01, tol=1e-6, max_iter=1000):
    """
    Implementación del Gradiente Descendente.
    
    f_grad: función que devuelve el gradiente.
    x0: punto inicial.
    lr: tasa de aprendizaje.
    tol: tolerancia de convergencia.
    max_iter: número máximo de iteraciones.
    """
    x = x0
    for i in range(max_iter):
        grad = f_grad(x)
        x_new = x - lr * grad
        
        if np.linalg.norm(grad) < tol:
            break
        x = x_new
        
    return x

# Ejemplo: minimizar f(x) = x^2 + 2x + 1
def grad_f(x):
    return 2*x + 2

x_min = gradient_descent(grad_f, x0=np.array([0.0]))
print("Mínimo encontrado en:", x_min)


# Definir variables simbólicas
x1, x2 = sp.symbols('x1 x2')

# Definir funciones residuales
r1 = x1**2 + x2 - 2
r2 = x1 + x2**2 - 2

# Vector de funciones
R = sp.Matrix([r1, r2])

# Calcular el Jacobiano automáticamente
J = R.jacobian([x1, x2])

# Convertir a funciones numéricas
J_func = sp.lambdify((x1, x2), J, 'numpy')
R_func = sp.lambdify((x1, x2), R, 'numpy')

# Evaluar en un punto
import numpy as np
x0 = np.array([1.0, 1.0])
print("Jacobiano en x0:\n", J_func(*x0))
print("Residuo en x0:\n", R_func(*x0))

def gauss_newton(J, r, x0, tol=1e-6, max_iter=100):
    """
    Implementación del método de Gauss-Newton.
    
    J: función que devuelve la matriz Jacobiana.
    r: función que devuelve el residuo.
    x0: punto inicial.
    tol: tolerancia.
    max_iter: número máximo de iteraciones.
    """
    x = x0
    for i in range(max_iter):
        J_eval = J(x)
        r_eval = r(x)
        
        delta_x = np.linalg.lstsq(J_eval, -r_eval, rcond=None)[0]
        x_new = x + delta_x
        
        if np.linalg.norm(delta_x) < tol:
            break
        x = x_new
        
    return x

# Ejemplo: ajuste de una función no lineal
def residual(x):
    return np.array([(x[0] + 2)**2 - 4])

def jacobian(x):
    return np.array([[2 * (x[0] + 2)]])

x_opt = gauss_newton(jacobian, residual, x0=np.array([0.0]))
print("Solución encontrada:", x_opt)


x1 = -1
x2 = 1
x3 = 1

y1 = 0
y2 = 1/2
y3 = -1/2