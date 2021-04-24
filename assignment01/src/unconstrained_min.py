import numpy as np 
from src import utils

def gradient_descent(f, x0, step_size, obj_tol, param_tol, max_iter):
    y0, grad0 = f(x0)
    path = [(x0, y0, grad0)]
    for i in range(max_iter):
        x1 = x0 - step_size * grad0
        y1, grad1 = f(x1)
        path.append((x1, y1, grad1))
        delta_x, delta_y = np.linalg.norm(x1-x0), np.abs(y1-y0) 
        utils.report(i, x0, y0, delta_x, delta_y)
        if delta_x < param_tol or delta_y < obj_tol:
            return path, True
        else:
            x0, y0, grad0 = x1, y1, grad1
    return path, False