import numpy as np

from VascularFlow.Elasticity.Beam import euler_bernoulli_transient
from VascularFlow.Flow.Flow import flow_rate
from VascularFlow.Flow.Pressure import pressure
from VascularFlow.Initialization.InitializeConstants import initialize_constants
from VascularFlow.Initialization.InitializeFlowArrays import initialize_flow_arrays

def two_way_coupled_fsi(
    nb_nodes: int,
    time_step_size: float,
    end_time: float,
    channel_aspect_ratio: float,
    reynolds_number: float,
    strouhal_number: float,
    fsi_parameter: float,
    relaxation_factor: float,
    inlet_flow_rate: float,
    h_n_1: np.array,
    h_n: np.array,
    h_star: np.array,
    h_new: np.array,
    q_n_1: np.array,
    q_n: np.array,
    q_star: np.array,
    q_new: np.array,
    p: np.array,
    p_inner: np.array,
):
    x_n = np.linspace(0, 1, nb_nodes)
    element_length = 1 / (len(x_n) - 1)

    time = 0
    outer_iteration_number = 0

    residual_values = []
    iteration_indices = []
    global_inner_counter = 0
    while time < end_time:
        time += time_step_size
        outer_iteration_number += 1
        inner_iteration_number = 0
        inner_residual = 1
        while inner_residual > 10e-08 and inner_iteration_number < 500:
            # pressure calculation
            p = pressure(
                x_n,
                element_length,
                time_step_size,
                channel_aspect_ratio,
                reynolds_number,
                strouhal_number,
                h_star,
                q_star,
                q_n,
                q_n_1,
            )

            # height calculation
            h_star = euler_bernoulli_transient(
                x_n,
                element_length,
                1,
                time_step_size,
                p,
                fsi_parameter,
                relaxation_factor,
                h_new,
            )

            # flow rate calculation
            q_star = flow_rate(
                x_n,
                element_length,
                time_step_size,
                strouhal_number,
                inlet_flow_rate,
                h_star,
                h_n,
                h_n_1,
            )

            # update inner iteration
            if max(abs(h_new - 1)) < 10e-20:
                inner_residual1 = max(abs(h_star - h_new)) / (max(abs(h_new)) + 10e-20)
            else:
                inner_residual1 = max(abs(h_star - h_new)) / max(abs(h_new))

            if max(abs(p_inner)) < 10e-20:
                inner_residual2 = max(abs(p - p_inner)) / (max(abs(p_inner)) + 10e-20)
            else:
                inner_residual2 = max(abs(p - p_inner)) / max(abs(p_inner))

            inner_residual = max(inner_residual1, inner_residual2)

            residual_values.append(inner_residual)
            iteration_indices.append(global_inner_counter)
            global_inner_counter += 1



            p_inner = p
            h_new = h_star
            q_new = q_star

            inner_iteration_number += 1


        print(f"Time: {time:.5f}, Inner Iteration: {inner_iteration_number}, Inner Residual: {inner_residual:.5e}", flush=True)

        h_n_1 = h_n
        h_n = h_new
        q_n_1 = q_n
        q_n = q_new

    return h_n, q_n, p, residual_values, iteration_indices

