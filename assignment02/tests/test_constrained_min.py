import numpy as np
import unittest
from src import constrained_min, utils
from tests import examples
import matplotlib.pyplot as plt
from functools import partial


class TestConstrainedMin(unittest.TestCase):
    def test_qp(self):
        ineq_constraints = [
            partial(examples.l, a=utils.col_v([-1, 0, 0])),
            partial(examples.l, a=utils.col_v([0, -1, 0])),
            partial(examples.l, a=utils.col_v([0, 0, -1])),
        ]
        eq_constraints_mat = np.array([[1, 1, 1]])
        eq_constraints_rhs = np.array([[1]])
        x0 = utils.col_v([0.1, 0.2, 0.7])
        path = constrained_min.interior_pt(examples.qp, ineq_constraints, 
            eq_constraints_mat, eq_constraints_rhs, x0)
        points = [p[0].T[0] for p in path]        
        outer_points = [p[0].T[0] for p in path if p[2] == 'outer']        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d', xlim=[0,1], ylim=[0,1], zlim=[0,1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.plot(xs=[p[0] for p in points], ys=[p[1] for p in points], zs=[p[2] for p in points])
        ax.scatter(xs=[p[0] for p in points[:1]], ys=[p[1] for p in points[:1]], zs=[p[2] for p in points[:1]],
            label=f'start ({points[0][0]:.2f}, {points[0][1]:.2f}, {points[0][2]:.2f})', color='green')
        ax.scatter(xs=[p[0] for p in points[-1:]], ys=[p[1] for p in points[-1:]], zs=[p[2] for p in points[-1:]],
            label=f'end ({points[-1][0]:.2f}, {points[-1][1]:.2f}, {points[-1][2]:.2f})', color='red')
        ax.scatter(xs=[p[0] for p in outer_points], ys=[p[1] for p in outer_points], zs=[p[2] for p in outer_points],
            label='outer_iteration', color='orange')
        plt.legend()
        plt.show()

    def test_lp(self):
        ineq_constraints = [
            partial(examples.l, a=utils.col_v([-1, -1]), r=1),
            partial(examples.l, a=utils.col_v([0, 1]), r=-1),
            partial(examples.l, a=utils.col_v([1, 0]), r=-2),
            partial(examples.l, a=utils.col_v([0, -1]), r=0),
        ]
        eq_constraints_mat = None
        eq_constraints_rhs = None
        x0 = utils.col_v([0.5, 0.75])
        f = partial(examples.l, a=utils.col_v([1, 1]))
        path = constrained_min.interior_pt(
            f, ineq_constraints, 
            eq_constraints_mat, eq_constraints_rhs, x0)
        points = [p[0].T[0] for p in path]
        zs = [f(p)[0] for p in points]
        outer_points = [p[0].T[0] for p in path if p[2] == 'outer']     
        outer_zs = [f(p)[0] for p in outer_points]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('objective')
        ax.plot(xs=[p[0] for p in points], ys=[p[1] for p in points], zs=zs)
        ax.scatter(xs=[p[0] for p in points[:1]], ys=[p[1] for p in points[:1]], zs=zs[:1],
            label=f'start ({points[0][0]:.2f}, {points[0][1]:.2f})', color='green')
        ax.scatter(xs=[p[0] for p in points[-1:]], ys=[p[1] for p in points[-1:]], zs=zs[-1:],
            label=f'end ({points[-1][0]:.2f}, {points[-1][1]:.2f})', color='red')
        ax.scatter(xs=[p[0] for p in outer_points[1:-1]], ys=[p[1] for p in outer_points[1:-1]], zs=outer_zs[1:-1],
            label='outer_iteration', color='orange')
        plt.legend()
        plt.show()
        


if __name__ == '__main__':
    unittest.main()