import numpy as np

from VascularFlow.Flow.Pressure import pressure
from VascularFlow.Elasticity.Beam import euler_bernoulli_transient
from VascularFlow.Flow.Flow import flow_rate
from VascularFlow.Initialization.InitializeConstants import initialize_constants
from VascularFlow.Initialization.InitializeFlowArrays import initialize_flow_arrays


def test_flow(plot=True):
    left = 0
    right = 1
    nb_nodes = 101
    x_n = np.linspace(left, right, 101)
    dx_e = (right - left) / (len(x_n) - 1)
    dt = 2.5e-3
    num_steps = 1
    # Initialize constants
    constants = initialize_constants()
    channel_aspect_ratio = constants["epsilon"]
    reynolds_number = constants["Re"]
    strouhal_number = constants["St"]
    fsi_parameter = constants["Beta"]
    relaxation_factor = constants["relax"]
    inlet_flow_rate = constants["q0"]

    # Initialize flow arrays
    flow_arrays = initialize_flow_arrays(nb_nodes)
    h_star = flow_arrays["h_star"]
    q_star = flow_arrays["q_star"]
    q_n = flow_arrays["q_n"]
    q_n1 = flow_arrays["q_n_1"]

    # pressure calculation
    pp = pressure(
        x_n,
        dx_e,
        dt,
        channel_aspect_ratio,
        reynolds_number,
        strouhal_number,
        h_star,
        q_star,
        q_n,
        q_n1,
    )
    # pp_interleaved = np.zeros(len(pp) * 2)
    # pp_interleaved[::2] = pp

    # Initialize flow arrays
    h_new = flow_arrays["h_new"]
    # h_new = np.concatenate([h_new, h_new])

    # channel height calculation
    channel_height = euler_bernoulli_transient(
        x_n,
        dx_e,
        num_steps,
        dt,
        pp,
        fsi_parameter,
        relaxation_factor,
        h_new,
    )

    # Initialize flow arrays
    h_n = flow_arrays["h_n"]
    h_n_1 = flow_arrays["h_n_1"]

    # flow rate calculation
    qq = flow_rate(
        x_n, dx_e, dt, strouhal_number, inlet_flow_rate, channel_height, h_n, h_n_1
    )
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1)
        ax[0].plot(x_n, pp)
        ax[1].plot(x_n, channel_height)
        ax[2].plot(x_n, qq)

        ax[0].set_xlabel("x")
        ax[0].set_ylabel("pressure")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("height")
        ax[2].set_xlabel("x")
        ax[2].set_ylabel("flow rate")
        plt.tight_layout()
        plt.show()
