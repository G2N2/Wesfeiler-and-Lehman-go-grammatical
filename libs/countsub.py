import libs.utils_matlang as mat
import numpy as np
import numpy.linalg as lin


def triangle(A):
    return np.trace(lin.matrix_power(A, 3))/6

def tailedtri(A):
    return (mat.one(A).T@(lin.matrix_power(A,3)*mat.diag(A@mat.one(A)-2))@mat.one(A))[0,0]/2

def fivecycle(A):
    a=A
    A2=a.dot(a)
    A3=A2.dot(a)
    A4=A3.dot(a)
    A5=A4.dot(a)
    one = mat.one(a)
    I = mat.diag(one)
    J = one@one.T - I
    r = (A5-2*A3*A2)*I-2*(np.diag((((a*(A2))@a)*J).sum(1)) - A3/2*I)
    r -= 2*(np.diag(a@((A3/2*I).sum(1)))-A3*I)
    return 1/10*r.sum()

def square(A):
    a=A
    A2=a.dot(a)
    A3=A2.dot(a)
    A4=A3.dot(a)
    one = mat.one(a)
    I = mat.diag(one)
    J = one@one.T - I
    return 1/8*(A4*I-A2*J-A2*A2*I).sum()

def trisquare(A):
    A2 = A@A
    return 1/4*((A2*A)*(A2*A-(A2*A>0))).sum()


def tailed4(A):
    a=A
    A2=a.dot(a)
    A3=A2.dot(a)
    A4=A3.dot(a)
    one = mat.one(a)
    I = mat.diag(one)
    J = one@one.T - I
    return 1/2*((A4*I-np.diag((A2*J).sum(1))-A2*A2*I)*(mat.diag(a.dot(one)-2))- np.diag((((a*(A2))*((a*A2)%2 ==0)).sum(1)))).sum()

