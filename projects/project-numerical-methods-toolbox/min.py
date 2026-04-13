
import numpy as np
import matplotlib.pyplot as plt

n = 100
num_experimentos = 5  # Número de veces que repetimos el experimento
semillas = [42, 123, 999, 723, 302]  # Lista de semillas para reproducibilidad

# Definir la función real
def f(x):
    return 2*x + 1

# Crear un rango de valores de x
x_values = np.linspace(0, 10, n)

plt.figure(figsize=(8, 6))  # Configurar el gráfico

for seed in semillas:
    np.random.seed(seed)  # Fijar la semilla
    
    # Generar ruido
    mu = 0
    sigma = np.sqrt(np.sqrt(2))
    epsilon = np.random.normal(loc=mu, scale=sigma, size=n)

    # Seleccionar 70 valores aleatorios sin reemplazo
    indices = np.random.choice(n, size=70, replace=False)
    samples = x_values[indices]
    epsilon_sampled = epsilon[indices]
    y_values_sampled = f(samples) + epsilon_sampled  

    # Cálculo de mínimos cuadrados
    A = np.vstack([samples, np.ones(len(samples))]).T
    m, b = np.linalg.lstsq(A, y_values_sampled, rcond=None)[0]

    # Graficar los puntos de datos
    plt.scatter(samples, y_values_sampled, color='red', alpha=0.6, label="Datos ajustados" if seed == semillas[0] else "")

    # Graficar la recta ajustada
    plt.plot(samples, m*samples + b, label=f"Semilla {seed}: y = {m:.2f}x + {b:.2f}")

# Graficar la función original
plt.plot(x_values, f(x_values), 'g--', label="Función real")

# Configuración del gráfico
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparación de ajustes con diferentes semillas")
plt.legend()
plt.grid(True)
plt.show()