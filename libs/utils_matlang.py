import numpy as np
import numpy.linalg as lin
from tqdm import tqdm

def one(e):
    if len(np.shape(e))== 1:
        n = np.shape(e)
        return np.ones(1)
    else:
        n,m = np.shape(e)
        return np.ones((n,1))


def diag(e):
    return np.diag(e[:,0])

def apply(f,M):
    return f(*M)


def scal(d):
    return np.array([d])

def trace(A):
    return (one(A).T@(A * diag(one(A)))@one(A))[0,0]

def adjacence(n):
    mat = np.random.random((n, n))
    mat = mat + mat.T
    A = (mat >= 1).astype(int)
    for i in range(A.shape[0]):
        A[i ,i] = 0
    return A

def ml1str(A,k):
    vect = ['1']
    for i in range(k):
        temp = []
        for v in vect:
            if v != '1':
                for w in vect:
                    if w != '1':
                        temp.append('diag('+v+')'+w)
            temp.append('A'+v)
        vect = temp + vect
        vect = list(set(vect))
    return vect

def ml1scal(A,N):
    vect = [one(A)]
    for i in tqdm(range(N)):
        temp = []
        for v in vect:
            temp.append(A@v)
            for w in vect:
                temp.append(diag(v)@w) 
        vect = vect + temp
    r = [1]
    for i in tqdm(range(len(vect))):
        for w in vect:
            r.append((vect[i].T@w)[0,0])
    r = list(set(r))
    return r

def ml3scal(A,N):
    vect = [A]
    for i in tqdm(range(N)):
        temp = []
        for v in vect:
            temp.append(diag(v@one(A) ))
            for w in vect:
                temp.append(v@w)
                temp.append(v*w)
        vect = vect + temp
    r = [1]
    for i in tqdm(range(len(vect))):
        for w in vect:
            r.append((one(A).T@vect[i]@one(A))[0,0])
    r = list(set(r))
    return r


def mot(X):
    C = X*lin.matrix_power(X,2)
    C = lin.matrix_power(C,2)
    H = C@one(X)
    H = H*H
    return one(X).T@H

def mot2(X):
    C = X*lin.matrix_power(X,2)
    C = C*C
    H = C@one(X)
    H = H*H
    return one(X).T@H









    