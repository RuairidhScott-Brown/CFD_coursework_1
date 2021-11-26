import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
import scipy
import pprint
import copy

import pprint


def LU_inplace(A): 
    """Compute the LU factorisation of A, using the in-place scheme so
    that the strictly lower triangular components of the array contain
    the strictly lower triangular components of L, and the upper
    triangular components of the array contain the upper triangular
    components of U.

    :param A: an mxm-dimensional numpy array

    """
    Matrix = A.copy()
    m,n = Matrix.shape
    
    for k in range(m-1):
        Matrix[k+1:,k] = Matrix[k+1:,k]/Matrix[k,k]
        Matrix[k+1:,k+1:] = Matrix[k+1:,k+1:] - np.outer(Matrix[k+1:,k], Matrix[k,k+1:])
    return Matrix



A = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)

m = LU_inplace(A)
U = np.triu(m)
L = np.tril(m, k=-1)
L = L + np.eye(L.shape[0])
print(U)
print(L)
print(L@U - A)


x = scipy.linalg.solve(A, [2, 4, -1])
x_dot = np.linalg.inv(L)@np.array([2, 4, -1], dtype=float)
x_dot = np.linalg.inv(U)@x_dot
print(x)
print(x_dot)
