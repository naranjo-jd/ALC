import numpy as np
import matplotlib.pyplot as plt


A1 = np.array([
    [2, 1],
    [1, 2]
])

A2 = np.array([
    [-3, 1],
    [1, 2]
])

def svd(A):
    SU, U = np.linalg.eig(A @ A.T)
    SV, V = np.linalg.eig(A.T @ A)
    indices_U = np.argsort(SU)[::-1]
    indices_V = np.argsort(SV)[::-1]
    U = U[:, indices_U]
    U = U / np.linalg.norm(U, axis=0)
    V = V[:, indices_V]
    V = V / np.linalg.norm(V, axis=0)
    S = np.zeros(A.shape)
    np.fill_diagonal(S, np.sqrt(SV[indices_V]))
    return U, S, V.T

U1, S1, VT1 = svd(A1)
U2, S2, VT2 = svd(A2)

# Assuming A1 is already defined
U1, S1, VT1 = np.linalg.svd(A1)

# Number of points
n = 100

# Generate angles equally spaced between 0 and 2*pi
angles = np.linspace(0, 2 * np.pi, n)

# Calculate x and y coordinates using cosine and sine
x = np.cos(angles)
y = np.sin(angles)

points = np.vstack((x, y))

# Multiply by U1, S1 (as a diagonal matrix), and VT1
VT_points = VT1.dot(points)
S_points = np.diag(S1).dot(VT_points)  # Correcting this by creating a diagonal matrix for S1
U_points = U1.dot(S_points)

# Create the 2x2 subplot grid
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Plot the original set of points in the first subplot (top-left)
axs[0, 0].scatter(points[0], points[1], color='b', label='Set 1')
axs[0, 0].set_title('Set 1')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
axs[0, 0].set_aspect('equal', adjustable='box')

# Plot the set of points after multiplying by U1 (left singular vectors) in the second subplot (top-right)
axs[0, 1].scatter(U_points[0], U_points[1], color='r', label='Set 2')
axs[0, 1].set_title('Set 2: U1 * Points')
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
axs[0, 1].set_aspect('equal', adjustable='box')

# Plot the set of points after multiplying by S1 (singular values) in the third subplot (bottom-left)
axs[1, 0].scatter(S_points[0], S_points[1], color='g', label='Set 3')
axs[1, 0].set_title('Set 3: S1 * Points')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')
axs[1, 0].set_aspect('equal', adjustable='box')

# Plot the set of points after multiplying by VT1 (right singular vectors) in the fourth subplot (bottom-right)
axs[1, 1].scatter(VT_points[0], VT_points[1], color='m', label='Set 4')
axs[1, 1].set_title('Set 4: VT1 * Points')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Y')
axs[1, 1].set_aspect('equal', adjustable='box')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()