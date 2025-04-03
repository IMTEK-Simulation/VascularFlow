"""
Numerical simulation for the coupled FSI equations.

Equations:
- Non-dimensionalized Navier–Stokes equations (conservation of momentum) → ∂p/∂x = -ReSt/H ∂Q/∂t - 6/5 Re/H ∂/∂x(Q2/H) - 12Q/H3
- Non-dimensionalized linear Euler–Bernoulli beam equation → ∂2w/∂t2 + ∂4w/∂x4 = p
- Non-dimensionalized Navier–Stokes equations (conservation of mass) → ∂Q/∂x = -St ∂H/∂t

Boundary conditions:
- zero pressure at the outlet of the channel.
- zero displacement and rotation at beam ends.
- Inlet area flow rate (equal to 1) at the inlet of the channel.

Numerical scheme:
- Finite element method
"""

import numpy as np

from VascularFlow.Coupling.FSIInnerLoop import inner_fsi_iteration


def two_way_coupled_fsi(
    mesh_nodes: np.ndarray,
    time_step_size: float,
    end_time: float,
    channel_aspect_ratio: float,
    reynolds_number: float,
    strouhal_number: float,
    fsi_parameter: float,
    relaxation_factor: float,
    inlet_flow_rate: float,
    inner_res: float,
    inner_it_number: int,
    inner_tolerance: float,
    h_n_1: np.array,
    h_n: np.array,
    h_star: np.array,
    h_new: np.array,
    q_n_1: np.array,
    q_n: np.array,
    q_star: np.array,
    q_new: np.array,
    p: np.array,
    p_new: np.array,
):
    """
    Simulate the fluid–structure interaction in a 1D channel with an elastic upper wall.
    This includes computing:
        - The pressure distribution along the channel
        - The area flow rate of the fluid
        - The displacement of the elastic wall caused by the fluid pressure

    Parameters
    ----------
    mesh_nodes : np.ndarray
        Global mesh node coordinates along the channel or beam.
    time_step_size : float
        Time step size of the coupled FSI equations.
    end_time : float
        End time of the coupled FSI equations.
    channel_aspect_ratio : float
        Channel’s aspect ratio (channel height / channel length).
    reynolds_number : float
        Reynolds number.
    strouhal_number : float
        Strouhal number.
    fsi_parameter : float
        Fluid-structure interaction parameter.
    relaxation_factor : float
        Under Relaxation factor (URF) ∈ (0,1].
    inlet_flow_rate : float
        Channel inlet flow rate.
    inner_res : float
        Inner residual number used in sub time steps.
    inner_it_number : int
        Inner iteration number used in sub time steps.
    inner_tolerance : float
        Inner tolerance used in sub time steps.
    h_n_1 : np.ndarray
        Channel height in previous time step.
    h_n : np.ndarray
        Channel height in current time step.
    h_star : np.ndarray
        Channel height in sub time step.
    h_new : np.ndarray
        Channel height in next time step.
    q_n_1 : np.ndarray
        Channel flow rate in previous time step.
    q_n : np.ndarray
        Channel flow rate in current time step.
    q_star : np.ndarray
        Channel flow rate in sub time step.
    q_new : np.ndarray
        Channel flow rate in next time step.
    p : np.ndarray
        Channel pressure at current time step.
    p_new : np.ndarray
        Channel pressure at next time step.

    Returns
    ----------
    h_n: np.ndarray
        Channel height at the end time of the simulation.
    q_n: np.ndarray
        Channel flow rate at the end time of the simulation.
    p: np.ndarray
        Channel pressure at the end time of the simulation.
    """

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
        while inner_residual > inner_res and inner_iteration_number < inner_it_number:
            # Update Channel height, flow rate, and pressure in sub time step
            h_star, q_star, p, inner_residual = inner_fsi_iteration(
                mesh_nodes,
                time_step_size,
                channel_aspect_ratio,
                reynolds_number,
                strouhal_number,
                fsi_parameter,
                relaxation_factor,
                inlet_flow_rate,
                inner_tolerance,
                h_n_1,
                h_n,
                h_star,
                h_new,
                q_n_1,
                q_n,
                q_star,
                p_new,
            )

            residual_values.append(inner_residual)
            iteration_indices.append(global_inner_counter)
            global_inner_counter += 1

            p_new = p
            h_new = h_star
            q_new = q_star

            inner_iteration_number += 1
            print(inner_iteration_number)
        print(
            f"Time: {time:.5f}, Inner Iteration: {inner_iteration_number}, Inner Residual: {inner_residual:.5e}",
            flush=True,
        )

        h_n_1 = h_n
        h_n = h_new
        q_n_1 = q_n
        q_n = q_new

    return h_n, q_n, p, residual_values, iteration_indices
