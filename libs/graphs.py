import numpy as np


def f(x):
    r = 0
    if x != 0:
        r = (1.*x)**(-1)
    return r
f = np.vectorize(f)

def Laplacian(A):
    D = np.diag(A.sum(0))
    return D - A

def Laplaciannormal(A):
    n,m = np.shape(A)
    D = np.diag(f((A.sum(axis = 1))**(.5)))
    return np.eye(n)- D@A@D

def Laplacianrw(A):
    n,m = np.shape(A)
    D = np.diag(f(A.sum(axis = 1)))
    return np.eye(n) - D@A