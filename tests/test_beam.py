import numpy as np

from VascularFlow.Elasticity.Beam import euler_bernoulli, euler_bernoulli_transient
from VascularFlow.Numerics.BasisFunctions import QuadraticBasis
from VascularFlow.Numerics.ElementMatrices import force_matrix


def test_euler_bernoulli_constant_load(plot=True):
    left = 0
    right = 1
    x_n = np.linspace(left, right, 11)
    dx_e = (right - left) / (len(x_n) - 1)
    p = np.ones(len(x_n) * 2)
    p[1::2] = 0

    lhs_matrix = euler_bernoulli(x_n, dx_e, p)[0]
    rhs_matrix = euler_bernoulli(x_n, dx_e, p)[1]
    det = np.linalg.det(lhs_matrix)



    w_g = euler_bernoulli(x_n, dx_e , p)[2]
    exact = (x_n ** 4 / 24) - (x_n ** 3 / 12) + (x_n ** 2 / 24)
    print(w_g.shape)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(x_n, w_g[::2])
        plt.plot(x_n, w_g[1::2])
        #plt.plot(x_n, exact, '*')
        #plt.plot(x, w_g.dot(QuadraticBasis().eval(x_g)))
        plt.xlabel('x')
        plt.ylabel('w')
        plt.show()


def test_euler_bernoulli_constant_load_transient(plot=True):
    left = 0
    right = 1
    x_n = np.linspace(left, right, 101)
    dx_e = (right - left) / (len(x_n) - 1)
    dt = 2.5e-03
    num_steps = 1
    p = (-12 * x_n + 12)
    #p_interleaved = np.zeros(len(p) * 2)
    #p_interleaved[::2] = p
    beta = 35156.24
    relaxation = 0.00003
    H_new = np.ones(len(x_n))

    channel_height = euler_bernoulli_transient(x_n, dx_e, num_steps, dt, p, beta, relaxation, H_new)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(x_n, channel_height)
        plt.xlabel('x')
        plt.ylabel('displacement')
        plt.show()
