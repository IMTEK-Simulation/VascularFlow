"""
1D numerical solution of the non-dimensionalized Navier–Stokes equations (conservation of mass)
to calculate the area flow rate through a channel with length 1, using the finite element method.

States:
- transient (∂q/∂x = -St ∂H/∂t)

Boundary conditions:
- Inlet area flow rate at the inlet of the channel

Basis function:
- Linear basis function
"""

import numpy as np
from scipy.sparse.linalg import spsolve

from VascularFlow.Numerics.Assembly import assemble_global_matrices
from VascularFlow.Numerics.BasisFunctions import LinearBasis
from VascularFlow.Numerics.ElementMatrices import (
    stiffness_matrix_first_derivative,
    load_vector,
)


def flow_rate(
    mesh_nodes: np.ndarray,
    time_step_size: float,
    st: float,
    inlet_flow_rate: float,
    h_star: np.ndarray,
    h_n: np.ndarray,
    h_n_1: np.ndarray,
):
    """
    Calculate the area flow rate at a given time step size through a channel with length 1, using the finite element method .

    Parameters
    ----------
    mesh_nodes : np.ndarray
        Global mesh node positions along the channel.
    time_step_size : float
        Time step size.
    st : float
        Strouhal number.
    inlet_flow_rate : float
        Inlet flow rate.
    h_star : np.ndarray
        Channel height in sub time step.
    h_n : np.ndarray
        Channel height in current time step.
    h_n_1 : np.ndarray
        Channel height in previous time step.

    Returns
    ----------
    area flow rate: np.ndarray
        area flow rate through the channel.

    """

    basis_function = LinearBasis()
    element_stiffness_matrix = stiffness_matrix_first_derivative
    element_load_vector = load_vector

    global_stiffness_matrix, global_load_vector = assemble_global_matrices(
        mesh_nodes,
        basis_function,
        element_stiffness_matrix,
        element_load_vector,
        3,
    )

    # Create the right hand side term (-St ∂H/∂t)
    first_term = (-st * (3 * h_star - 4 * h_n + h_n_1)) / (2 * time_step_size)

    lhs = global_stiffness_matrix
    rhs = global_load_vector * first_term

    # Add boundary condition (inlet area flow rate at the inlet of the channel)
    lhs[0] = 0
    lhs[0, 0] = 1
    rhs[0] = inlet_flow_rate

    channel_flow_rate = spsolve(lhs, rhs)

    return channel_flow_rate
