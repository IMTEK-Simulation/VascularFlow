import numpy as np

from VascularFlow.Elasticity.Beam import euler_bernoulli
from VascularFlow.Numerics.BasisFunctions import QuadraticBasis


def test_euler_bernoulli_constant_load(plot=True):
    left = 0
    right = 1
    x_g = np.linspace(left, right, 7)
    x_n = x_g[::2]
    p_g = np.ones_like(x_g)
    w_g = euler_bernoulli(x_n, p_g)

    if plot:
        import matplotlib.pyplot as plt
        x = np.linspace(left, right, 101)
        plt.plot(x_g, w_g, 'kx')
        plt.plot(x, QuadraticBasis().interpolate(x_n, w_g, x), 'k-')
        plt.xlabel('x')
        plt.ylabel('w')
        plt.show()
