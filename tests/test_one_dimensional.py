
from VascularFlow.Coupling.OneDimensional import two_way_coupled_fsi
from VascularFlow.Initialization.InitializeConstants import initialize_constants
from VascularFlow.Initialization.InitializeFlowArrays import initialize_flow_arrays


def test_two_way_coupled_fsi(plot=True):
    nb_nodes = 200
    time_step_size = 2.5e-03
    nb_time_steps = 8000
    end_time = nb_time_steps * 2.5e-03

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
    print(final_solution[0])
    print(final_solution[1])
    print(final_solution[2])
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.semilogy(final_solution[4], final_solution[3])
        plt.xlabel("Cumulative Inner Iteration Count")
        plt.ylabel("Inner Residual (log scale)")
        plt.title("Inner Residual vs Iterations")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("Inner Residual vs Iterations.png")
        plt.show()


if __name__ == "__main__":
    test_two_way_coupled_fsi()
