import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data.csv")

# Número total de datos
num_datos = data.shape[0]
print("Numero total de datos:", num_datos)

# Mezclar los índices de los datos
np.random.seed(42)
indices = np.random.permutation(num_datos)

# Separar en 80% entrenamiento y 20% prueba
train_size = int(0.8 * num_datos)  # 80% del total

train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Crear los conjuntos de entrenamiento y prueba
train_data = data.iloc[train_indices]
test_data = data.iloc[test_indices]

# Extraer las variables predictoras y la variable dependiente del conjunto de prueba
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

def polynomial_design_matrix(X, degree):
    """
    Genera la matriz de diseño para regresión polinómica usando solo numpy.
    
    Parameters:
    X : ndarray de forma (m, n) con m muestras y n características.
    degree : int, grado máximo del polinomio.
    
    Returns:
    X_poly : ndarray de forma (m, k) con los términos polinómicos hasta el grado especificado.
    """
    m, n = X.shape
    X_poly = np.ones((m, 1))  # Incluir término de sesgo
    
    for d in range(1, degree + 1):
        for i in range(n):
            X_poly = np.hstack((X_poly, X[:, i:i+1]**d))
    
    return X_poly

def mean_squared_error(y_true, y_pred):
    """Calcula el error cuadrático medio (MSE)."""
    return np.mean((y_true - y_pred) ** 2)


n = 5 # Grado maximo del polinomio

# Diccionario para almacenar los modelos
models_ec = {}

# Iterar sobre los grados del polinomio (de 1 a n)
for i in range(1, n+1):
    # Construir la matriz de diseño para entrenamiento y prueba
    X_poly_train = polynomial_design_matrix(X_train, i)
    X_poly_test = polynomial_design_matrix(X_test, i)

    # Aplicar la ecuación normal para calcular los coeficientes theta
    theta = np.linalg.pinv(X_poly_train.T @ X_poly_train) @ X_poly_train.T @ y_train

    # Calcular el número de condición
    cond_number = np.linalg.cond(X_poly_train.T @ X_poly_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = X_poly_test @ theta

    # Calcular el MSE en el conjunto de prueba
    mse = mean_squared_error(y_test, y_pred)

    # Almacenar el modelo en el diccionario
    models_ec[i] = {
        "theta": theta,
        "cond_number": cond_number,
        "MSE": mse
    }

    # Imprimir los resultados
    print(f"Grado {i}: Número de condición = {cond_number}, MSE = {mse}")

# Encontrar el modelo con el menor MSE
best_degree = min(models_ec, key=lambda k: models_ec[k]["MSE"])
print(f"\nEl mejor modelo es el de grado {best_degree} con MSE = {models_ec[best_degree]['MSE']}")



# Gradiente

def gradient_descent(X, y, alpha=0.01, tol=1e-6, max_iterations=1000):
    """
    Gradiente descendente con parada temprana por convergencia.
    
    Parámetros:
    X : ndarray (m, n) - Matriz de diseño
    y : ndarray (m, 1) - Variable objetivo
    alpha : float - Tasa de aprendizaje
    tol : float - Tolerancia para detectar convergencia
    max_iterations : int - Número máximo de iteraciones
    
    Retorna:
    theta : ndarray (n, 1) - Coeficientes ajustados
    history : lista - Historial de costos
    """
    m, n = X.shape
    theta = np.zeros((n, 1))  # Inicializar en ceros
    history = []  # Para almacenar el historial de costos
    
    prev_cost = float('inf')
    
    for i in range(max_iterations):
        y_pred = X @ theta  # Predicciones
        error = y_pred - y  # Error
        gradient = (1/m) * (X.T @ error)  # Gradiente
        theta -= alpha * gradient  # Actualización de theta
        
        # Calcular el costo actual
        cost = mean_squared_error(y, y_pred)
        history.append(cost)
        
        # Criterio de parada: si el costo cambia muy poco, detener iteraciones
        if abs(prev_cost - cost) < tol:
            print(f"Convergencia alcanzada en {i} iteraciones.")
            break
        
        prev_cost = cost
    
    return theta, history

n = 5  # Grado máximo del polinomio
alpha = 0.01  # Tasa de aprendizaje
max_iterations = 1000  # Número de iteraciones

# Diccionario para almacenar los modelos
models_gd = {}

# Iterar sobre los grados del polinomio (de 1 a n)
for i in range(1, n+1):
    # Construir la matriz de diseño para entrenamiento y prueba
    X_poly_train = polynomial_design_matrix(X_train, i)
    X_poly_test = polynomial_design_matrix(X_test, i)

    # Ajustar el modelo con gradiente descendente
    theta, cost_history = gradient_descent(X_poly_train, y_train, alpha, tol=1e-6, max_iterations=max_iterations)

    # Hacer predicciones en el conjunto de prueba
    y_pred = X_poly_test @ theta

    # Calcular el MSE en el conjunto de prueba
    mse = mean_squared_error(y_test, y_pred)

    # Almacenar el modelo en el diccionario
    models_gd[i] = {
        "theta": theta,
        "MSE": mse,
        "cost_history": cost_history
    }

    # Imprimir los resultados
    print(f"Grado {i}: MSE = {mse:.4f}")

# Encontrar el modelo con el menor MSE
best_degree = min(models_gd, key=lambda k: models_gd[k]["MSE"])
print(f"\nEl mejor modelo es el de grado {best_degree} con MSE = {models_gd[best_degree]['MSE']:.4f}")

# Diccionario para almacenar resultados
comparison = {}


# Evaluar modelos
for i in range(1, n+1):
    # Obtener theta de ambos métodos
    theta_gd = models_gd[i]["theta"]  # Gradiente descendente
    theta_ne = models_ec[i]["theta"]

    # Predecir
    y_pred_gd = X_poly_test @ theta_gd
    y_pred_ne = X_poly_test @ theta_ne

    # Calcular MSE
    mse_gd = mean_squared_error(y_test, y_pred_gd)
    mse_ne = mean_squared_error(y_test, y_pred_ne)

    # Guardar resultados
    comparison[i] = {"MSE_gradiente": mse_gd, "MSE_normal": mse_ne}

    print(f"Grado {i}: MSE GD = {mse_gd:.4f}, MSE NE = {mse_ne:.4f}")