import numpy as np

def generate_matrices(points, degree=1):
    """
    Generates the design matrix A and vector b for least squares regression.

    Parameters:
        points (array-like): Nx2 array or list of (x, y) points.
        degree (int): Degree of the polynomial fit (default is 1 for linear regression).

    Returns:
        A (numpy array): The design matrix.
        b (numpy array): The target vector.
    """
    # Convert to NumPy array if not already
    points = np.asarray(points)  # Ensures input works for lists and np arrays

    # Extract x and y values
    x_vals = points[:, 0]  # First column (x-values)
    b = points[:, 1]       # Second column (y-values)

    # Construct the design matrix A (Vandermonde matrix for polynomial fitting)
    A = np.vander(x_vals, N=degree+1, increasing=True)

    return A, b

def least_squares(A, b):
    """
    Solves the least squares problem Ax = b using the normal equations.
    
    Parameters:
        A (numpy.ndarray): The input matrix.
        b (numpy.ndarray): The right-hand side vector.
    
    Returns:
        numpy.ndarray: The least squares solution.
    """
    return np.linalg.pinv(A) @ b

def gradient_descent(A, b, lr=0.01, tol=1e-6, max_iter=1000):
    """
    Solves the least squares problem Ax = b using gradient descent.
    
    Parameters:
        A (numpy.ndarray): The input matrix.
        b (numpy.ndarray): The right-hand side vector.
        lr (float, optional): Learning rate (default is 0.01).
        tol (float, optional): Convergence tolerance (default is 1e-6).
        max_iter (int, optional): Maximum number of iterations (default is 1000).
    
    Returns:
        numpy.ndarray: The approximate solution.
    """
    x = np.zeros(A.shape[1])
    for _ in range(max_iter):
        gradient = A.T @ (A @ x - b)
        if np.linalg.norm(gradient) < tol:
            break
        x -= lr * gradient
    return x

