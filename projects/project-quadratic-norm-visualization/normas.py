import matplotlib.pyplot as plt
import numpy as np


def norm_A(X, A):
    """Computes the norm of vector X with respect to matrix A."""
    return np.sqrt(X.T @ A @ X)


def plot_norm_circle(A, center=np.array([0, 0]), radius=1, resolution=100):
    """Plots the set of points at a fixed norm-distance from the center under the quadratic norm."""
    theta = np.linspace(0, 2 * np.pi, resolution)

    # Generate points on the unit circle
    unit_circle = np.array([np.cos(theta), np.sin(theta)])

    # Transform the unit circle using A^(-1/2) to account for the quadratic form
    eigvals, eigvecs = np.linalg.eigh(A)  # Eigen decomposition
    A_sqrt_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T  # A^(-1/2)

    transformed_points = radius * A_sqrt_inv @ unit_circle  # Apply transformation
    transformed_points[0] += center[0]  # Translate
    transformed_points[1] += center[1]

    # Plot the transformed unit circle
    plt.figure(figsize=(6, 6))
    plt.plot(transformed_points[0], transformed_points[1], label=f"Norm-{radius} Contour")
    plt.scatter(*center, color="red", label="Center")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title("Quadratic Norm Contour")
    plt.axis("equal")
    plt.show()


# Example usage
A = np.array([[3, 1], [1, 2]])  # Example positive definite matrix
plot_norm_circle(A, center=np.array([1, 1]), radius=2)
