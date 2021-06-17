import numpy as np
import unittest
from src import unconstrained_min, utils
from tests import examples

class TestUnconstrainedMin(unittest.TestCase):

    def test_quad_min(self):
        for f, title in [(examples.q1, "q1"), (examples.q2, "q2"), (examples.q3, "q3")]:
            path, done = unconstrained_min.line_search(f, np.array([1,1]), 1e-12, 1e-8, 100, 'gd')
            utils.plot_it(f, path, title, done)

    def test_rosenbrock_min(self):
        for f, title in [(examples.rosen, "rosenbrock")]:
            path, done = unconstrained_min.line_search(f, np.array([2,2]), 1e-7, 1e-8, 10000, 'gd')
            utils.plot_it(f, path, title, done)

    def test_lin_min(self):
        for f, title in [(examples.l1, "linear")]:
            path, done = unconstrained_min.line_search(f, np.array([1,1]), 1e-12, 1e-8, 100, 'gd')
            utils.plot_it(f, path, title, done)


if __name__ == '__main__':
    unittest.main()