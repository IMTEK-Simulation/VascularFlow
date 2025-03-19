import numpy as np

from VascularFlow.Elasticity.Beam import euler_bernoulli


def test_euler_bernoulli_constant_load():
    x_g = np.linspace(0, 1, 21)
    x_n = x_g[::2]
    p_g = np.ones_like(x_g)
    w_g = euler_bernoulli(x_n, p_g)

    print(w_g)
