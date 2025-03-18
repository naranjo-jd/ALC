import numpy as np

def condition_number(A, p=2):
    """
    Computes the condition number of a matrix A.
    
    Parameters:
        A (numpy.ndarray): The input matrix.
        p (int or str, optional): The norm type (default is 2, the spectral condition number).
                                  Other options include:
                                  - 'fro' for Frobenius norm
                                  - 1 for maximum column sum norm
                                  - np.inf for maximum row sum norm
                                  - Other p-norms supported by numpy.linalg.norm
    
    Returns:
        float: The condition number of A.
    """
    try:
        return np.linalg.cond(A, p)
    except np.linalg.LinAlgError:
        return float('inf')  # If the matrix is singular, return infinity
    
def squared_residuals(y_true, y_fitted):
    """
    Computes the Sum of Squared Residuals (SSR).
    SSR = sum((y_i - fitted(y_i))^2)

    Parameters:
        y_true (array-like): Actual values.
        y_fitted (array-like): Fitted (predicted) values.

    Returns:
        float: Sum of squared residuals.
    """
    residuals = y_true - y_fitted
    return np.sum(residuals**2)

def residual_norm(y_true, y_fitted):
    """
    Computes the Residual Norm (L2 norm of residuals).
    Residual Norm = sqrt(sum((y_i - fitted(y_i))^2))

    Parameters:
        y_true (array-like): Actual values.
        y_fitted (array-like): Fitted (predicted) values.

    Returns:
        float: Residual norm.
    """
    residuals = y_true - y_fitted
    return np.linalg.norm(residuals, 2)