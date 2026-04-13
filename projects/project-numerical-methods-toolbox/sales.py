import numpy as np
import matplotlib.pyplot as plt
import models

# Data: [city, price, sales/week]
data = np.array([
    [1,  0.59, 3980],
    [2,  0.80, 2200],
    [3,  0.95, 1850],
    [4,  0.45, 6100],
    [5,  0.79, 2100],
    [6,  0.99, 1700],
    [7,  0.90, 2000],
    [8,  0.65, 4200],
    [9,  0.79, 2440],
    [10, 0.69, 3300],
    [11, 0.79, 2300],
    [12, 0.49, 6000],
    [13, 1.09, 1190],
    [14, 0.95, 1960],
    [15, 0.79, 2760],
    [16, 0.65, 4330],
    [17, 0.45, 6960],
    [18, 0.60, 4160],
    [19, 0.89, 1990],
    [20, 0.79, 2860],
    [21, 0.99, 1920],
    [22, 0.85, 2160]
])

# ======= Linear Regression (y = ax + b) =======
A1, b1 = models.generate_matrices(data[:, 1:], degree=1)  # Only Price & Sales columns

# Modelos
theta_ls1 = models.least_squares(A1, b1)
theta_gd1 = models.gradient_descent(A1, b1, lr=0.01, max_iter=1000)

# Valores de prediccion
y_ls1 = A1 @ theta_ls1
y_gd1 = A1 @ theta_gd1

# ======= Plot Results =======
x = data[:, 1]
y = data[:, 2]

plt.scatter(x, y, color='blue', label="Data (Price vs Sales)")
plt.plot(x, y_ls1, color='red', label="Least Squares Fit")
plt.plot(x, y_gd1, color='green', linestyle="--", label="Gradient Descent Fit")

plt.xlabel("Price")
plt.ylabel("Sales per week")
plt.title("Linear Model: S = c1 + c2*P")
plt.legend()
plt.show()

# 2) Regresion cuadratica

A2, b2 = models.generate_matrices(data[:, 1:], degree=2)

theta_ls2 = models.least_squares(A2,b2)
theta_gd2 = models.gradient_descent(A2, b2, lr=0.01, max_iter=1000)

y_ls2 = A2 @ theta_ls2
y_gd2 = A2 @ theta_gd2

# Sort x (prices) and re-order everything accordingly
sorted_indices = np.argsort(x)  # Get sorted indices
x_sorted = x[sorted_indices]    # Sort x values
y_ls2_sorted = y_ls2[sorted_indices]  # Sort least squares predictions
y_gd2_sorted = y_gd2[sorted_indices]  # Sort gradient descent predictions

# Plot data and quadratic fits
plt.scatter(x, y, color='blue', label="Data (Price vs Sales)")
plt.plot(x_sorted, y_ls2_sorted, color='red', label="Least Squares Fit")
plt.plot(x_sorted, y_gd2_sorted, color='green', linestyle="--", label="Gradient Descent Fit")

plt.xlabel("Price")
plt.ylabel("Sales per week")
plt.title("Quadratic Model: S = c1 + c2*P + c3*P^2")
plt.legend()
plt.show()

print("Least Squares Solution:", theta_ls2)
print("Gradient Descent Solution:", theta_gd2)
print("Difference:", np.linalg.norm(theta_ls2 - theta_gd2))