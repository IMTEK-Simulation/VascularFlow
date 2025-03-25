import numpy as np

from VascularFlow.Flow.Pressure import pressure
from VascularFlow.Elasticity.Beam import euler_bernoulli_transient
from VascularFlow.Flow.Flow import flow_rate


def test_flow(plot=True):
    left = 0
    right = 1
    x_n = np.linspace(left, right, 200)
    dx_e = (right - left) / (len(x_n) - 1)
    dt = 2.5e-3
    num_steps = 1
    eps = 0.02
    re = 7.5
    st = 0.68
    beta = 35156.24
    relaxation = 0.00003
    inlet_flow_rate = 1

    # pressure calculation
    Hstar = np.ones(len(x_n))
    Qstar = np.ones(len(x_n))
    Q_n = np.ones(len(x_n))
    Q_n1 = np.ones(len(x_n))
    pp = pressure(x_n, dx_e, dt, eps, re, st, Hstar, Qstar, Q_n, Q_n1)[2]
    pp_interleaved = np.zeros(len(pp) * 2)
    pp_interleaved[::2] = pp

    # channel height calculation
    H_new = np.ones(len(x_n) * 2)
    channel_height = euler_bernoulli_transient(x_n, dx_e, num_steps, dt, pp_interleaved, beta, relaxation, H_new)[1]

    # flow rate calculation

    H_n = np.ones(len(x_n))
    H_n1 = np.ones(len(x_n))

    qq = flow_rate(x_n, dx_e, dt, st, inlet_flow_rate, channel_height[::2], H_n, H_n1)

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3,1)
        ax[0].plot(x_n, pp)
        ax[1].plot(x_n, channel_height[::2])
        ax[2].plot(x_n, qq)

        ax[0].set_xlabel('x')
        ax[0].set_ylabel('pressure')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('height')
        ax[2].set_xlabel('x')
        ax[2].set_ylabel('flow rate')
        plt.tight_layout()
        plt.show()