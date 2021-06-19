import numpy as np
import unittest
from src import unconstrained_min, utils
from tests import examples
import matplotlib.pyplot as plt

class TestUnconstrainedMin(unittest.TestCase):

    def test_rosenbrock_min(self):
        for f, title in [(examples.rosen, "rosenbrock")]:
            print(title)
            _, axis = plt.subplots(1, 3)
            _, axis2 = plt.subplots(1, 1)
            plt.suptitle(title)
            for idx, method in enumerate(['gd', 'nt', 'bfgs',]):
                print(method)
                path, done = unconstrained_min.line_search(f, np.array([2,2]), 1e-7, 1e-8, 10000, method)
                utils.plot_it(axis[idx], f, path, method, done)
                utils.plot_convergence(axis2, path, method)
            axis2.legend()
            plt.savefig(fname=title + '.png')
            plt.show()

    def test_quad_min(self):
        for f, title in [(examples.q1, "q1"), (examples.q2, "q2"), (examples.q3, "q3")]:
            print(title)
            _, axis = plt.subplots(1, 3)
            _, axis2 = plt.subplots(1, 1)
            plt.suptitle(title)
            for idx, method in enumerate(['gd', 'nt', 'bfgs',]):
                print(method)
                path, done = unconstrained_min.line_search(f, np.array([1,1]), 1e-12, 1e-8, 100, method)
                utils.plot_it(axis[idx], f, path, method, done)
                utils.plot_convergence(axis2, path, method)
            axis2.legend()
            plt.savefig(fname=title + '.png')
            plt.show()


if __name__ == '__main__':
    unittest.main()