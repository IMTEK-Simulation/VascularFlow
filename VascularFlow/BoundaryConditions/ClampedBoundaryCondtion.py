"""
Clamped boundary conditions for an elastic beam with length l, used in 1D numerical solution of Euler Bernoulli beam theory.

Boundary conditions:
- zero displacement and rotation at beam ends (i.e., fully fixed).

"""

import numpy as np


def clamped_boundary_condition(
    global_matrix: np.array,
    global_vector: np.array,
):
    """
    Apply clamped (fully fixed) boundary conditions to the global stiffness matrix and load vector of the beam problem.

    Parameters
    ----------
    global_matrix: np.array
        Global stiffness (or mass) matrix before applying boundary conditions.
    global_vector: np.array
        Global load vector before applying boundary conditions.

    Returns
    ----------
    global_matrix: np.array
        Modified stiffness matrix after applying clamped boundary conditions.
    global_vector: np.array
        Modified load vector after applying clamped boundary conditions.
    """
    n = global_matrix.shape[0]

    # Fix the first two and last two degrees of freedom (u0, u0', uL', uL)
    clamped_dofs = [0, 1, n - 2, n - 1]

    for dof in clamped_dofs:
        global_matrix[dof, :] = 0
        global_matrix[dof, dof] = 1
        global_vector[dof] = 0

    return global_matrix, global_vector
