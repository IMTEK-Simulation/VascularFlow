import numpy as np
from VascularFlow.Elasticity.Beam import euler_bernoulli_steady
from VascularFlow.Flow.Pressure import pressure_steady_state

def steady_state_fsi(
        mesh_nodes: np.ndarray,
        channel_aspect_ratio: float,
        reynolds_number: float,
        fsi_parameter: float,
        relaxation_factor: float,
        inner_tolerance: float,
        residual_number,
        iteration_number,
        h_star: np.ndarray,
        h_new: np.ndarray,
        p_new: np.ndarray,
):
    residual = 1
    iteration = 0

    while residual>residual_number and iteration < iteration_number:
        # pressure calculation
        p = pressure_steady_state(
            mesh_nodes,
            channel_aspect_ratio,
            reynolds_number,
            h_star,
        )


        # beam displacement calculation
        w = euler_bernoulli_steady(
            mesh_nodes,
            p,
        )

        # channel height calculation
        h_star = 1 + fsi_parameter * w
        h_star = relaxation_factor * h_star + (1 - relaxation_factor) * h_new

        # update
        if np.max(np.abs(h_new - 1)) < inner_tolerance:
            residual_h = np.max(np.abs(h_star - h_new)) / (
                    np.max(np.abs(h_new)) + inner_tolerance
            )
        else:
            residual_h = np.max(np.abs(h_star - h_new)) / np.max(np.abs(h_new))

        if np.max(np.abs(p_new)) < inner_tolerance:
            residual_p = np.max(np.abs(p - p_new)) / (
                    np.max(np.abs(p_new)) + inner_tolerance
            )
        else:
            residual_p = np.max(np.abs(p - p_new)) / np.max(np.abs(p_new))

        inner_residual = max(residual_h, residual_p)

        p_new = p
        h_new = h_star

        iteration += 1

    return p_new, h_new

