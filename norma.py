import numpy as np
import matplotlib.pyplot as plt

def norm_A(X, A):
    """Computes the norm of vector X with respect to matrix A."""
    return np.sqrt(X.T @ A @ X)

def plot_norm_circle(A, center=np.array([0, 0]), radius=1, resolution=100):
    """Plots the set of points at a fixed norm-distance from the center under the quadratic norm."""
    theta = np.linspace(0, 2 * np.pi, resolution)
    
    # Generate points on the unit circle
    unit_circle = np.array([np.cos(theta), np.sin(theta)])

    # Eigen decomposition of A
    eigvals, eigvecs = np.linalg.eigh(A)  # Eigenvalues and eigenvectors
    
    # A^(-1/2)
    A_sqrt_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
    
    # Transform the unit circle using A^(-1/2) to account for the quadratic form
    transformed_points = radius * A_sqrt_inv @ unit_circle  # Apply transformation
    transformed_points[0] += center[0]  # Translate
    transformed_points[1] += center[1]

    # Plot the transformed unit circle
    plt.figure(figsize=(6, 6))
    plt.plot(transformed_points[0], transformed_points[1], label=f"Norm-{radius} Contour")
    plt.scatter(*center, color='red', label="Center")
    
    # Plot eigenvectors scaled by eigenvalues
    for i in range(len(eigvals)):
        eigenvector = eigvecs[:, i]
        eigenvalue = eigvals[i]
        plt.quiver(*center, *eigenvector, color=f'C{i+2}', scale=eigenvalue*2, label=f"Eigenvector {i+1} (λ={eigenvalue:.2f})")
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    
    # Mover la leyenda fuera de la gráfica
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Coloca la leyenda a la derecha fuera del gráfico
    
    plt.title("Quadratic Norm Contour with Eigenvectors")
    plt.axis("equal")
    plt.tight_layout()  # Ajusta el diseño para evitar superposiciones
    plt.show()

# Example Usage
A = np.array([[3, 1], [1, 2]])  # Example positive definite matrix
plot_norm_circle(A, center=np.array([0, 0]), radius=2)