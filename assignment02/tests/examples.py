import numpy as np

def q(x, Q):
    return x.T @ Q @ x, 2 * Q @ x, 2 * Q

def q1(x):
    Q = np.array([[1, 0], [0, 1]])
    return q(x, Q)

def q2(x):
    Q = np.array([[5, 0], [0, 1]])
    return q(x, Q)

def q3(x):
    A = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    B = np.array([[5, 0], [0, 1]])
    Q = A.T @ B @ A
    return q(x, Q)

def rosen(x):
    x1, x2 = x
    grad = np.array([-400*x1*x2 + 400*x1**3 + 2*x1 - 2, 200*x2 - 200*x1**2])
    hess = np.array([
        [-400*x2+1200*x1**2+2, -400*x1],
        [-400*x1, 200]
    ])
    return 100*(x2-x1**2)**2 + (1-x1)**2, grad, hess

def l(x, a):
    return a.T @ x, a, 0

def l1(x):
    a = np.array([25, 8])
    return l(x, a)

