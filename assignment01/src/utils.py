import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def report(i, x1, y1, delta_x, delta_y):
    print(f"i={i}: f({x1})={y1}, dx={delta_x}, dy={delta_y}")

def plot_it(f, path, step_size, title, done):
    steps = len(path)    
    if len(path) > 300:
        path = path[:30] + path[-30:]
    min_i, max_i = min(x[0] for x, _, _ in path), max(x[0] for x, _, _ in path)
    min_j, max_j = min(x[1] for x, _, _ in path), max(x[1] for x, _, _ in path)
    di, dj = 2*(max_i-min_i), 2*(max_j-min_j)
    i = np.arange(min_i-di, max_i+di, di/100)
    j = np.arange(min_j-dj, max_j+dj, dj/100)
    I, J = np.meshgrid(i, j)
    X = np.vstack((np.ravel(I), np.ravel(J))).T
    Z = np.array([f(exam)[0] for exam in X])
    Z = Z.reshape(I.shape)
    fig, ax = plt.subplots()
    CS = ax.contour(I, J, Z)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title(title + f", converged={done}, steps={steps}")
    ax.scatter([x[0] for x, _, _ in path], [x[1] for x, _, _ in path])
    for x, y, grad in path:
        ax.arrow(x[0], x[1], -grad[0]*step_size*0.5, -grad[1]*step_size*0.5, head_width=abs(grad[0]*step_size*0.3))
    last_x1, last_x2 = path[-1][0][0], path[-1][0][1]
    ax.plot(last_x1, last_x2, 'ro', label=f"{last_x1}, {last_x2}")
    plt.legend()        
    plt.savefig(fname=title + '.png')
    plt.show()


