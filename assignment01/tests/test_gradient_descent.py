import numpy as np
import unittest
from src import unconstrained_min, utils
from tests import examples

class TestGradientDescent(unittest.TestCase):

    def test_quad_min(self):
        for f, title in [(examples.q1, "q1"), (examples.q2, "q2"), (examples.q3, "q3")]:
            step_size = 0.2
            path, done = unconstrained_min.gradient_descent(f, np.array([1,1]), step_size, 1e-12, 1e-8, 100)
            utils.plot_it(f, path, step_size, title, done)

    def test_rosenbrock_min(self):
        for f, title in [(examples.rosen, "rosenbrock")]:
            step_size = 0.001
            path, done = unconstrained_min.gradient_descent(f, np.array([2,2]), step_size, 1e-7, 1e-8, 10000)
            utils.plot_it(f, path, step_size, title, done)

    def test_lin_min(self):
        for f, title in [(examples.l1, "linear")]:
            step_size = 0.2
            path, done = unconstrained_min.gradient_descent(f, np.array([1,1]), step_size, 1e-12, 1e-8, 100)
            utils.plot_it(f, path, step_size, title, done)


if __name__ == '__main__':
    unittest.main()