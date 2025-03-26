import numpy as np

from VascularFlow.Coupling.OneDimensional import two_way_coupled_fsi
from VascularFlow.Initialization.InitializeConstants import initialize_constants
from VascularFlow.Initialization.InitializeFlowArrays import initialize_flow_arrays


def test_two_way_coupled_fsi(plot=True):
    nb_nodes = 11
    time_step_size = 2.5e-03
    end_time = 1

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
    h_n_1 = flow_arrays["h_n_1"]
    h_n = flow_arrays["h_n"]
    h_star = flow_arrays["h_star"]
    h_new = flow_arrays["h_new"]
    q_n_1 = flow_arrays["q_n_1"]
    q_n = flow_arrays["q_n"]
    q_star = flow_arrays["q_star"]
    q_new = flow_arrays["q_new"]
    p = flow_arrays["p"]
    p_inner = flow_arrays["p_inner"]


    final_solution = two_way_coupled_fsi(
        nb_nodes,
        time_step_size,
        end_time,
        channel_aspect_ratio,
        reynolds_number,
        strouhal_number,
        fsi_parameter,
        relaxation_factor,
        inlet_flow_rate,
        h_n_1,
        h_n,
        h_star,
        h_new,
        q_n_1,
        q_n,
        q_star,
        q_new,
        p,
        p_inner,
    )

    if plot:
        import matplotlib.pyplot as plt
        x = np.linspace(0, 1, nb_nodes)
        plt.plot(x, final_solution)
        plt.xlabel('x')
        plt.ylabel('channel height')
        plt.show()

