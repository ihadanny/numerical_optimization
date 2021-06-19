import numpy as np 
from src import utils

def line_search(f, x0, obj_tol, param_tol, max_iter, 
    dir_selection_method, init_step_len=1.0, slope_ratio=1e-4, back_track_factor=0.2):
    y0, grad0, hess0 = f(x0)
    Bk = hess0
    path = [(x0, y0, grad0, hess0)]
    for i in range(max_iter):
        if dir_selection_method == "gd":
            pk = -grad0
        elif dir_selection_method == "nt":
            pk = newton_dir(grad0, hess0)
        elif dir_selection_method == "bfgs":
            pk = newton_dir(grad0, Bk)
        alpha = find_wolfe_step_size(f, x0, pk, grad0, init_step_len, slope_ratio, back_track_factor)
        x1 = x0 + alpha * pk
        y1, grad1, hess1 = f(x1)
        delta_x, delta_y = np.linalg.norm(x1-x0), np.abs(y1-y0)
        utils.report(i, pk, x0, y0, delta_x, delta_y, alpha)
        if dir_selection_method == "bfgs":
            Bk = update_bfgs(Bk, x0, x1, grad0, grad1)
        if delta_x < param_tol or delta_y < obj_tol:
            return path, True
        else:
            path.append((x1, y1, grad1, hess1))
            x0, y0, grad0, hess0 = x1, y1, grad1, hess1
    return path, False

def update_bfgs(Bk, x0, x1, grad0, grad1):
    sk = (x1 - x0).reshape(-1, 1)
    yk = (grad1 - grad0).reshape(-1, 1)
    return Bk - Bk @ sk @ sk.T @ Bk.T / (sk.T @ Bk @ sk) + (yk @ yk.T) / (yk.T @ sk)

def newton_dir(grad0, hess0):
    return np.linalg.solve(hess0, -grad0)

def find_wolfe_step_size(f, xk, pk, grad0, init_step_len, slope_ratio, back_track_factor):
    alpha = init_step_len
    while f(xk + alpha * pk)[0] > f(xk)[0] + slope_ratio * alpha * grad0.T @ pk:
        alpha = back_track_factor * alpha
    return alpha
