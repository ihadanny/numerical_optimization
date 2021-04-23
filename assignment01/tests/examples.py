import numpy as np

def q(x, Q):
    return np.matmul(np.matmul(np.transpose(x), Q), x)

def q1(x):
    Q = np.array([[1, 0], [0, 1]])
    return q(x, Q)

def q2(x):
    Q = np.array([[5, 0], [0, 1]])
    return q(x, Q)

def q3(x):
    A = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    B = np.array([[5, 0], [0, 1]])
    Q = np.matmul(np.matmul(np.transpose(A), B), A)
    return q(x, Q)
