import numpy as np
import scipy.linalg

np.set_printoptions(suppress=True, precision=4)  # precision work

def lu_factorization_partial(A):
    """
    Performs LU factorization with partial pivoting of a square matrix A.
    
    Parameters:
        A (numpy.ndarray): The input square matrix.
    
    Returns:
        tuple: (P, L, U) where P is the permutation matrix, L is the lower triangular matrix,
               and U is the upper triangular matrix.
    """
    P, L, U = scipy.linalg.lu(A)
    return P, L, U

def lu_factorization_full(A):
    """
    Performs LU factorization with full pivoting of a square matrix A.
    
    Parameters:
        A (numpy.ndarray): The input square matrix.
    
    Returns:
        tuple: (P, Q, L, U) where P and Q are permutation matrices, L is the lower triangular matrix,
               and U is the upper triangular matrix.
    """
    m, n = A.shape
    P = np.eye(m)
    Q = np.eye(n)
    L = np.zeros_like(A, dtype=np.float64)
    U = A.copy()
    
    for i in range(min(m, n)):
        row, col = np.unravel_index(np.abs(U[i:, i:]).argmax(), U[i:, i:].shape)
        row += i
        col += i
        
        U[[i, row], :] = U[[row, i], :]
        P[[i, row], :] = P[[row, i], :]
        U[:, [i, col]] = U[:, [col, i]]
        Q[:, [i, col]] = Q[:, [col, i]]
        
        L[i, i] = 1.0
        for j in range(i+1, m):
            L[j, i] = U[j, i] / U[i, i]
            U[j, :] -= L[j, i] * U[i, :]
    
    return P, Q, L, U

def qr_decomposition(A):
    """
    Performs QR decomposition of a matrix A.
    
    Parameters:
        A (numpy.ndarray): The input matrix.
    
    Returns:
        tuple: (Q, R) where Q is an orthogonal matrix and R is an upper triangular matrix.
    """
    Q, R = np.linalg.qr(A)
    return Q, R

def cgs(A):
    m, n = A.shape
    Q = np.zeros([m,n], dtype=np.float64)
    R = np.zeros([n,n], dtype=np.float64)
    for j in range(n):
        v = A[:,j]
        for i in range(j):
            R[i,j] = np.dot(Q[:,i], A[:,j])
            v = v - (R[i,j] * Q[:,i])
        R[j,j] = np.linalg.norm(v)
        Q[:, j] = v / R[j,j]
    return Q, R

def mgs(A):
    V = A.copy()
    m, n = A.shape
    Q = np.zeros([m,n], dtype=np.float64)
    R = np.zeros([n,n], dtype=np.float64)
    for i in range(n):
        R[i,i] = np.linalg.norm(V[:,i])
        Q[:,i] = V[:,i] / R[i,i]
        for j in range(i, n):
            R[i,j] = np.dot(Q[:,i],V[:,j])
            V[:,j] = V[:,j] - R[i,j]*Q[:,i]
    return Q, R