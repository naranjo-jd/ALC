import numpy as np
import sympy as sp
import matplotlib as plt

X = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 1, 1.5, 3, 3.5, 5, 6],
    [1, 2, 3, 1.5, 5, 1.5, 3, 1]
] 

X = np.array(X).T

T = [0, 0, 0, 0, 1, 1, 1, 1]


def L(X):
    w = W[1:]
    w0 = W[0]
    return w.T @ X + w0


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# 1. Dataset de prueba (2 clases, 2 características)
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, random_state=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 2. Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 3. Regresión logística con descenso de gradiente
def logistic_regression(X, y, lr=0.1, n_iter=1000):
    m, n = X.shape
    X_b = np.hstack([X, np.ones((m, 1))])  # Añadimos columna de bias
    W = np.zeros(n + 1)

    for _ in range(n_iter):
        z = X_b @ W
        p = sigmoid(z)
        gradient = (1/m) * (X_b.T @ (p - y))
        W -= lr * gradient
    return W

# 4. Entrenamiento
W_log = logistic_regression(X, y)

# 5. Visualización
def plot_decision_boundary(X, y, W, title):
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
    x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    y_vals = -(W[0]*x_vals + W[2]) / W[1]
    plt.plot(x_vals, y_vals, 'k--')
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.show()

plot_decision_boundary(X, y, W_log, "Regresión Logística")

# 1. Agregamos columna de bias
m, n = X.shape
X_b = np.hstack([X, np.ones((m, 1))])

# 2. Regresión lineal usando ecuación normal
W_lin = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

# 3. Clasificación con umbral
def predict_linear(X, W):
    return (X @ W >= 0.5).astype(int)

# 4. Visualización
plot_decision_boundary(X, y, W_lin, "Regresión Lineal (Clasificación)")