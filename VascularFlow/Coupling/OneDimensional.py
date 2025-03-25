import numpy as np

from VascularFlow.Elasticity.Beam import euler_bernoulli_transient
from VascularFlow.Flow.Flow import flow_rate
from VascularFlow.Flow.Pressure import pressure

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
    while time < end_time:
        time += time_step_size
        outer_iteration_number += 1
        ###############################################################################################################
        inner_iteration_number = 0
        inner_residual_number = 1
        while inner_residual_number > 10e-06 and inner_iteration_number<20000:
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

            channel_pressure_solid_equation = np.zeros(len(p) * 2)
            channel_pressure_solid_equation[::2] = p

            # height calculation
            h_star = euler_bernoulli_transient(
                x_n,
                element_length,
                0,
                time_step_size,
                channel_pressure_solid_equation,
                fsi_parameter,
                relaxation_factor,
                h_new,
            )[1]

            #flow rate calculation
            q_star = flow_rate(
                x_n,
                element_length,
                time_step_size,
                strouhal_number,
                inlet_flow_rate,
                h_star[::2],
                h_n,
                h_n_1,
            )

            # update inner iteration
            if max(abs(h_new - 1)) < 1e-16:
                inner_res1 = max(abs(h_star - h_new)) / (max(abs(h_new)) + 1e-16)
            else:
                inner_res1 = max(abs(h_star - h_new)) / max(abs(h_new))

            if max(abs(p_inner)) < 1e-16:
                inner_res2 = max(abs(p - p_inner)) / (max(abs(p_inner)) + 1e-16)
            else:
                inner_res2 = max(abs(p - p_inner)) / max(abs(p_inner))

            inner_res = max(inner_res1, inner_res2)
            inner_resold = inner_res

            p_inner = p
            h_new = h_star
            inner_iteration_number += 1

        # residuals
        res1 = np.linalg.norm(abs(h_new - h_n)) / np.sqrt(np.size(h_new))
        res2 = np.max(abs(h_new - h_n))
        print(time, res1, res2)


        h_n_1 = h_n
        h_n = h_new
        q_n_1 = q_n
        q_n = q_new







