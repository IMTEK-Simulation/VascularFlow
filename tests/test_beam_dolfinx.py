import numpy as np

from VascularFlow.Elasticity.BeamDolfinx import euler_bernoulli_transient_dolfinx

from VascularFlow.Initialization.InitializeConstants import initialize_constants
from VascularFlow.Initialization.InitializeFlowArrays import initialize_flow_arrays

def test_euler_bernoulli_transient_dolfinx(plot=True):
    nb_nodes = 500
    x = np.linspace(0, 1, nb_nodes + 1)
    time_step_size = 2.5e-03
    nb_time_steps = 800
    end_time = nb_time_steps * time_step_size


    # Initialize constants
    constants = initialize_constants()
    fsi_parameter = constants["Beta"]
    relaxation_factor = constants["relax"]

    # Initialize flow arrays
    #flow_arrays = initialize_flow_arrays(nb_nodes)
    #h_new = flow_arrays["h_new"]
    h_new = np.ones(nb_nodes+1)

    pressure = np.ones(nb_nodes + 1)
    #pressure = (-10 * x) + 10

    displacement = euler_bernoulli_transient_dolfinx(
        nb_nodes,
        time_step_size,
        nb_time_steps,
        pressure,
        fsi_parameter,
        relaxation_factor,
        h_new,
    )

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(x, displacement)
        plt.xlabel("x")
        plt.ylabel("displacement")
        plt.grid(True)
        plt.show()