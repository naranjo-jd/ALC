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
models = {}

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
    models[i] = {
        "theta": theta,
        "cond_number": cond_number,
        "MSE": mse
    }

    # Imprimir los resultados
    print(f"Grado {i}: Número de condición = {cond_number}, MSE = {mse}")

# Encontrar el modelo con el menor MSE
best_degree = min(models, key=lambda k: models[k]["MSE"])
print(f"\nEl mejor modelo es el de grado {best_degree} con MSE = {models[best_degree]['MSE']}")



# Gradiente



def compute_cost(X, y, theta):
    """
    Calcula el error cuadrático medio (MSE) como función de costo.
    """
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Aplica gradiente descendente para minimizar la función de costo.
    
    Parameters:
        X : ndarray -> Matriz de diseño (muestras x características)
        y : ndarray -> Vector de etiquetas (m, 1)
        theta : ndarray -> Vector de coeficientes (n+1, 1)
        alpha : float -> Tasa de aprendizaje
        num_iters : int -> Número de iteraciones

    Returns:
        theta : ndarray -> Coeficientes ajustados
        J_history : list -> Historial de la función de costo
    """
    m = len(y)
    J_history = []

    for _ in range(num_iters):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1 / m) * (X.T @ errors)
        theta -= alpha * gradient  # Actualizar los coeficientes

        J_history.append(compute_cost(X, y, theta))

    return theta, J_history

# 🔹 Datos de ejemplo
X = np.array([[1], [2], [3], [4], [5]])  # Característica
y = np.array([[2], [3], [5], [7], [11]])  # Etiqueta

degree = 2  # Grado del polinomio
X_poly = polynomial_design_matrix(X, degree)  # Expandir características

# 🔹 Inicializar parámetros
theta = np.zeros((X_poly.shape[1], 1))
alpha = 0.01  # Tasa de aprendizaje
num_iters = 1000  # Iteraciones

# 🔹 Aplicar gradiente descendente
theta_final, J_history = gradient_descent(X_poly, y, theta, alpha, num_iters)

print("Coeficientes finales:\n", theta_final)

def mean_squared_error(y_true, y_pred):
    """Calcula el error cuadrático medio (MSE)."""
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, y, theta, alpha, num_iters):
    """Aplica gradiente descendente para minimizar la función de costo."""
    m = len(y)
    J_history = []

    for _ in range(num_iters):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1 / m) * (X.T @ errors)
        theta -= alpha * gradient  # Actualizar coeficientes

        J_history.append(mean_squared_error(y, predictions))

    return theta, J_history

# 🔹 Configuración
n = 5  # Grado máximo del polinomio
alpha = 0.01  # Tasa de aprendizaje
num_iters = 1000  # Iteraciones del gradiente descendente

# 🔹 Diccionario para almacenar los modelos
models = {}

# 🔹 Extraer variables predictoras y dependientes
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)

# 🔹 Entrenamiento y evaluación de modelos con diferentes grados
for i in range(1, n + 1):
    # Expandir características para entrenamiento y prueba
    X_poly_train = polynomial_design_matrix(X_train, i)
    X_poly_test = polynomial_design_matrix(X_test, i)

    # Inicializar los coeficientes en ceros
    theta = np.zeros((X_poly_train.shape[1], 1))

    # Aplicar gradiente descendente
    theta_final, J_history = gradient_descent(X_poly_train, y_train, theta, alpha, num_iters)

    # Hacer predicciones en el conjunto de prueba
    y_pred = X_poly_test @ theta_final

    # Calcular el MSE en prueba
    mse = mean_squared_error(y_test, y_pred)

    # Almacenar el modelo
    models[i] = {
        "theta": theta_final,
        "MSE": mse
    }

    # Imprimir resultados
    print(f"Grado {i}: MSE en prueba = {mse}")

# 🔹 Determinar el mejor modelo
best_degree = min(models, key=lambda k: models[k]["MSE"])
print(f"\nEl mejor modelo es el de grado {best_degree} con MSE = {models[best_degree]['MSE']}")