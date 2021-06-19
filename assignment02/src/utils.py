import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def report(i, pk, x1, y1, delta_x, delta_y, alpha):
    print(f"i={i}: pk={pk}, f({x1})={y1}, dx={delta_x}, dy={delta_y}, alpha={alpha}")

def plot_it(ax, f, path, title, done):
    steps = len(path)    
    if len(path) > 300:
        path = path[:30] + path[-30:]
    min_i, max_i = min(x[0] for x, _, _, _ in path), max(x[0] for x, _, _, _ in path)
    min_j, max_j = min(x[1] for x, _, _, _ in path), max(x[1] for x, _, _, _ in path)
    di, dj = 2*(max_i-min_i), 2*(max_j-min_j)
    i = np.arange(min_i-di, max_i+di, di/100)
    j = np.arange(min_j-dj, max_j+dj, dj/100)
    I, J = np.meshgrid(i, j)
    X = np.vstack((np.ravel(I), np.ravel(J))).T
    Z = np.array([f(exam)[0] for exam in X])
    Z = Z.reshape(I.shape)
    CS = ax.contour(I, J, Z)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title(title + f", converged={done}, steps={steps}")
    ax.scatter([x[0] for x, _, _, _ in path], [x[1] for x, _, _, _ in path])
    for i, (x, y, grad, hess) in enumerate(path[:-1]):
        xn, _, _, _ = path[i+1]
        step_size = np.sqrt((xn[0]-x[0])**2 + (xn[1]-x[1])**2)
        ax.arrow(x[0], x[1], xn[0]-x[0], xn[1]-x[1], length_includes_head=True, head_width=step_size*0.1)
    last_x1, last_x2 = path[-1][0][0], path[-1][0][1]
    ax.plot(last_x1, last_x2, 'ro', label=f"{last_x1}, {last_x2}")
    ax.legend()        

def plot_convergence(ax, path, method):
    ax.plot([y for _, y, _, _ in path], label=method, marker='X')
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective")
    ax.set_yscale('log')

