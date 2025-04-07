
# Definir variables
x, y = sp.symbols('x y')

# Definir círculos (centro y radio)
circles = [(-1, 0, 1), (1, 1.5, 1.5), (1, -1.5, 1.5)]

# Definir función de error como matriz simbólica
F = sp.Matrix([sp.sqrt((x - xc)**2 + (y - yc)**2) - r for xc, yc, r in circles])

# Encontrar el punto más cercano
x0 = np.array([0, 0], dtype=float)
x_opt = gauss_newton(F, [x, y], x0)

print(f"Punto más cercano (Gauss-Newton): ({x_opt[0]:.6f}, {x_opt[1]:.6f})")


# Definir la función de error como suma de cuadrados de las diferencias de radios
f = sp.Rational(1, 2) * sum((sp.sqrt((x - xc)**2 + (y - yc)**2) - r)**2 for xc, yc, r in circles)

x_opt = gradient_descent(f, [x, y], x0)

print(f"Punto más cercano (Gradiente Descendente): ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
