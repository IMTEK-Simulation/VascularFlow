"""
1D numerical solution of the non-dimensionalized Navier–Stokes equations (conservation of momentum)
to calculate the pressure through a channel with length 1, using the finite element method.

States:
- steady (∂p/∂x = - 6/5 Re/H ∂/∂x(1/H) - 12/H3)
- transient (∂p/∂x = -ReSt/H ∂Q/∂t - 6/5 Re/H ∂/∂x(Q2/H) - 12Q/H3)

Boundary conditions:
- zero pressure at the outlet of the channel

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
from VascularFlow.Numerics.ArrayFirstDerivative import array_first_derivative


def pressure(
    mesh_nodes: np.ndarray,
    time_step_size: float,
    eps: float,
    re: float,
    st: float,
    h_star: np.ndarray,
    q_star: np.ndarray,
    q_n: np.ndarray,
    q_n_1: np.ndarray,
):
    """
    Calculate the pressure at a given time step size through a channel with length 1, using the finite element method.

    Parameters
    ----------
    mesh_nodes : np.ndarray
        Global mesh node positions along the channel.
    time_step_size : float
        Time step size.
    eps : float
        Channel’s aspect ratio (channel height/channel length).
    re : float
        Reynolds number.
    st : float
        Strouhal number.
    h_star : np.ndarray
        Channel height in sub time step.
    q_star : np.ndarray
        channel area flow rate in sub time step.
    q_n : np.ndarray
        channel area flow rate in current time step.
    q_n_1 : np.ndarray
        channel area flow rate in previous time step.

    Returns
    ----------
    pressure : np.ndarray
        Pressure through the channel

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

    # create the right hand side terms (-ReSt/H ∂Q/∂t, 6/5 Re/H ∂/∂x(Q2/H), 12Q/H3)
    first_term = (
        -((eps * re * st) / h_star)
        * (3 * q_star - 4 * q_n + q_n_1)
        / (2 * time_step_size)
    )
    second_term = (
        -1.2
        * re
        * (1 / h_star)
        * array_first_derivative(q_star**2 / h_star, mesh_nodes)
    )
    third_term = -12 * q_star / (h_star**3)

    lhs = global_stiffness_matrix
    rhs = global_load_vector * (first_term + second_term + third_term)

    # Add boundary condition (zero pressure at the outlet of the channel)
    lhs[-1] = 0
    lhs[-1, -1] = 1
    rhs[-1] = 0

    # Solve system
    channel_pressure = spsolve(lhs, rhs)

    return channel_pressure


def pressure_steady_state(
    mesh_nodes: np.ndarray,
    eps: float,
    re: float,
    h_star: np.ndarray,
):
    """
    Calculate the steady state pressure through a channel with length 1, using the finite element method.
    """
    basis_function = LinearBasis()
    element_stiffness_matrix = stiffness_matrix_first_derivative
    element_load_vector = load_vector
    global_stiffness_matrix, global_load_vector = assemble_global_matrices(
        mesh_nodes, basis_function, element_stiffness_matrix, element_load_vector, 3
    )
    # create the right hand side terms (6/5 Re/H ∂/∂x(1/H), 12/H3)
    first_term = (
        -1.2 * eps * re * (1 / h_star) * array_first_derivative(1 / h_star, mesh_nodes)
    )
    second_term = -12 / (h_star**3)

    lhs = global_stiffness_matrix
    rhs = global_load_vector * (first_term + second_term)

    # Add boundary condition (zero pressure at the outlet of the channel)
    lhs[-1] = 0
    lhs[-1, -1] = 1
    rhs[-1] = 0

    # Solve system
    channel_pressure_steady_state = spsolve(lhs, rhs)

    return channel_pressure_steady_state#, lhs, rhs
