import numpy as np
from VascularFlow.Flow.Pressure import pressure

def test_pressure(plot=True):
    left = 0
    right = 1
    x_n = np.linspace(left, right, 11)
    dx_e = (right - left) / (len(x_n) - 1)
    dt = 2.5e-3
    eps = 0.02
    re = 7.5
    st = 0.68
    Hstar = np.ones(len(x_n))
    Qstar = np.ones(len(x_n))
    Q_n = np.ones(len(x_n))
    Q_n1 = np.ones(len(x_n))

    pp = pressure(x_n, dx_e, dt, eps, re, st, Hstar, Qstar, Q_n, Q_n1)
    print(pp.shape)
    print(pp)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(x_n, pp, label="Pressure")
        plt.xlabel("x")
        plt.ylabel("pressure")
        plt.show()