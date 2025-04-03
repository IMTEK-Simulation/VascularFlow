import numpy as np
from VascularFlow.Flow.Pressure import pressure
from VascularFlow.Elasticity.Beam import euler_bernoulli_transient
from VascularFlow.Flow.Flow import flow_rate


def inner_fsi_iteration(
    mesh_nodes: np.ndarray,
    time_step_size: float,
    channel_aspect_ratio: float,
    reynolds_number: float,
    strouhal_number: float,
    fsi_parameter: float,
    relaxation_factor: float,
    inlet_flow_rate: float,
    inner_tolerance: float,
    h_n_1: np.ndarray,
    h_n: np.ndarray,
    h_star: np.ndarray,
    h_new: np.ndarray,
    q_n_1: np.ndarray,
    q_n: np.ndarray,
    q_star: np.ndarray,
    p_new: np.ndarray,
):
    """
    Perform a single inner iteration for the coupled 1D fluid–structure interaction (FSI) problem.

    This function updates:
        - Pressure via conservation of momentum (Navier–Stokes)
        - Channel wall displacement via Euler–Bernoulli beam theory
        - Area flow rate via conservation of mass (Navier–Stokes)

    It also computes the residuals to monitor convergence within the inner loop.

    Parameters
    mesh_nodes : np.ndarray
        Global mesh node coordinates along the channel or beam.
    time_step_size : float
        Time step size of the coupled FSI equations.
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
    inner_tolerance : float
        Inner tolerance used in sub time steps
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
    p_new : np.ndarray
        Channel pressure at next time step.

    Returns
    -------
    h_star : np.ndarray
        Updated channel height.
    q_star : np.ndarray
        Updated flow rate.
    p : np.ndarray
        Updated pressure distribution.
    residual : float
        Maximum residual computed across variables.
    """

    # pressure calculation
    p = pressure(
        mesh_nodes,
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
        mesh_nodes,
        1,
        time_step_size,
        p,
        fsi_parameter,
        relaxation_factor,
        h_new,
    )
    # flow rate calculation
    q_star = flow_rate(
        mesh_nodes,
        time_step_size,
        strouhal_number,
        inlet_flow_rate,
        h_star,
        h_n,
        h_n_1,
    )
    # update inner iteration
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

    return h_star, q_star, p, inner_residual
