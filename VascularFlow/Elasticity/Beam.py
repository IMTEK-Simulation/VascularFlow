"""
1D numerical solution of the non-dimensionalized linear Euler–Bernoulli beam theory
for an elastic beam of length 1, using the finite element method.

States:
- steady (∂4w/∂x4 = q)
- transient (∂2w/∂t2 + ∂4w/∂x4 = q)

Boundary conditions:
- zero displacement and rotation at beam ends

Basis function:
- Hermite basis function
"""

import numpy as np
from scipy.sparse.linalg import spsolve



from VascularFlow.Numerics.Assembly import assemble_global_matrices
from VascularFlow.Numerics.BasisFunctions import HermiteBasis
from VascularFlow.Numerics.ElementMatrices import (
    stiffness_matrix_fourth_derivative,
    mass_matrix_fourth_derivatives,
    load_vector,
)
from VascularFlow.BoundaryConditions.ClampedBoundaryCondtion import (
    clamped_boundary_condition,
)


def euler_bernoulli_steady(
    mesh_nodes: np.array,
    q: np.array,
):
    """
    Calculates the deflection (w) and rotation (∂w/∂x) of a clamped beam in steady state.

    Parameters
    ----------
    mesh_nodes : np.array
        Global mesh node positions along the beam.
    q : np.ndarray
        Distributed load for each nodal position along the beam.

    Returns
    -------
    Deflection_n : np.ndarray
        The deflection of the beam for each nodal position along the beam.
    Rotation_n : np.ndarray
        The rotation of the beam for each nodal position along the beam.
    """
    basis_function = HermiteBasis()
    element_stiffness_matrix = stiffness_matrix_fourth_derivative
    element_load_vector = load_vector
    global_stiffness_matrix, global_load_vector = assemble_global_matrices(
        mesh_nodes,
        basis_function,
        element_stiffness_matrix,
        element_load_vector,
        3,
    )

    q_interleaved = np.zeros(len(q) * 2)
    q_interleaved[::2] = q

    lhs = global_stiffness_matrix
    rhs = global_load_vector * q_interleaved

    # Add boundary conditions
    lhs_bc, rhs_bc = clamped_boundary_condition(lhs.copy(), rhs.copy())

    # Solve system
    solution = spsolve(lhs_bc, rhs_bc)
    displacement = solution[::2]
    # rotation = solution[1::2]
    return displacement


def euler_bernoulli_transient(
    mesh_nodes: np.array,
    nb_time_steps: int,
    time_step_size: float,
    q,
    beta,
    relaxation_factor,
    h_new,
):
    """
    Calculates the deflection (w) and rotation (∂w/∂x) of a clamped beam in transient state.

    Parameters
    ----------
    mesh_nodes : np.array
        Global mesh node positions along the beam.
    nb_time_steps : int
        Number of time steps.
    time_step_size : float
        Time step size.
    q : np.ndarray
        Distributed load for each nodal position along the beam.
    beta : float
        Fluid-structure interaction parameter to update channel height from displacement (H = 1 + β w).
    relaxation_factor : float
        Under Relaxation factor (URF) ∈ (0,1], used to achieve numerically stable results .
    h_new : np.ndarray
        Channel height in next time step used to under-relax calculated channel height (H = URF * H * (1 - URF) * H_new).

    Returns
    ----------
    Deflection_n : np.ndarray
        The deflection of the beam for each nodal position along the beam.
    Rotation_n : np.ndarray
        The rotation of the beam for each nodal position along the beam.
    """
    basis_function = HermiteBasis()
    element_stiffness_matrix = stiffness_matrix_fourth_derivative
    element_mass_matrix = mass_matrix_fourth_derivatives
    element_load_vector = load_vector

    global_stiffness_matrix, global_vector = assemble_global_matrices(
        mesh_nodes,
        basis_function,
        element_stiffness_matrix,
        element_load_vector,
        3,
    )
    global_mass_matrix, _ = assemble_global_matrices(
        mesh_nodes, basis_function, element_mass_matrix, element_load_vector, 3
    )

    lhs = global_mass_matrix + (time_step_size**2) * global_stiffness_matrix

    q_interleaved = np.zeros(len(q) * 2)
    q_interleaved[::2] = q
    global_load_vector = global_vector * q_interleaved

    # Add boundary conditions
    rhs_dummy = np.zeros(lhs.shape[0])
    lhs_bc, _ = clamped_boundary_condition(lhs.copy(), rhs_dummy)

    # Initialized arrays for displacement in current and previous time step.
    w_n = np.zeros(2 * len(mesh_nodes))
    w_n_1 = np.zeros(2 * len(mesh_nodes))
    for n in range(nb_time_steps):

        f1 = (time_step_size**2) * global_load_vector
        f2 = 2 * (global_mass_matrix @ w_n)
        f3 = global_mass_matrix @ w_n_1

        rhs = f1 + f2 - f3

        # Add boundary condition
        lhs_dummy = np.zeros((len(rhs), len(rhs)))
        _, rhs_bc = clamped_boundary_condition(lhs_dummy, rhs.copy())
        w_new = spsolve(lhs_bc, rhs_bc)

        w_n_1 = w_n
        w_n = w_new

    h_new_interleaved = np.zeros(len(h_new) * 2)
    h_new_interleaved[::2] = h_new
    channel_height = 1 + (beta * w_new)
    channel_height = (
        relaxation_factor * channel_height + (1 - relaxation_factor) * h_new_interleaved
    )

    return channel_height[::2]
