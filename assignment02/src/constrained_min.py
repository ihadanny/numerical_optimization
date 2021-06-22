import numpy as np 
from src import utils
from src.unconstrained_min import find_wolfe_step_size, newton_dir
EPS = 0.00001

def barrier_constraint(constraint, x0):
    # see http://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/15-barr-method-scribed.pdf
    y0, grad0, hess0 = constraint(x0)
    y_b = np.log(-y0)
    grad_b = -grad0 / y0
    hess_b = grad0 @ grad0.T / (y0*y0) - hess0 / y0
    return y_b, grad_b, hess_b

def barrier_f(t, f, ineq_constraints, x0):
    # all operators are linear
    y0, grad0, hess0 = f(x0)
    y0, grad0, hess0 = t*y0, t*grad0, t*hess0
    for constraint in ineq_constraints:
        yi, gradi, hessi = barrier_constraint(constraint, x0)
        y0, grad0, hess0 = y0+yi, grad0+gradi, hess0+hessi
    return y0, grad0, hess0

def interior_pt(f, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0,
    init_step_len=1.0, slope_ratio=1e-4, back_track_factor=0.2):
    m = len(ineq_constraints)
    t = 1
    print(f"m = {m}, {m / t}")
    path = []
    while m / t > EPS:
        # min t*func + phi(x), s.t. Ax=b
        print(f"outer_loop t = {t}")
        lambda_x = None
        iters = 0 
        while lambda_x is None or 0.5 * lambda_x > EPS:
            y0, grad0, hess0 = barrier_f(t, f, ineq_constraints, x0)
            path.append((x0, y0, 'inner'))
            if eq_constraints_mat is not None:
                pk = constrained_newton_dir(eq_constraints_mat, grad0, hess0)
            else:
                pk = newton_dir(grad0, hess0)
            lambda_x = np.asscalar(pk.T @ hess0 @ pk)
            #alpha = 1.0
            alpha = find_wolfe_step_size(f, x0, pk, grad0, init_step_len, slope_ratio, back_track_factor)
            print(f"inner_loop lambda={lambda_x} alpha={alpha}")            
            x0 = x0 + alpha * pk
            iters += 1
            if iters > 50 or alpha < EPS:
                break
        t *= 2
        path.append((x0, y0, 'outer'))
    return path

def constrained_newton_dir(A, grad, hess):
    # Newton direction
    # hess @ p + A.T @ w = -grad
    # A @ p = 0
    m, n = A.shape
    M = np.block([[hess, A.T],[A, np.zeros((m, m))]])
    b = np.vstack((-grad, np.zeros((m, 1))))
    v = np.linalg.solve(M, b)
    return v[:n]
