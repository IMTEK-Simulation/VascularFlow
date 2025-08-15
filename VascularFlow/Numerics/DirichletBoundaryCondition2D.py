# -----------------------------------------------------------------------------
# dirichlet_bc
# -----------------------------------------------------------------------------
# In finite element methods (FEM), Dirichlet boundary conditions enforce that
# the solution takes fixed values at specific degrees of freedom (DOFs).
# This function modifies the global system of equations [A]{u} = {L} to enforce
# these conditions by overwriting the Galerkin equations for the constrained DOFs.
#
# The algorithm works as follows:
#   1. For each DOF in a boundary group:
#      - Zero out the corresponding row in the system matrix A.
#      - Set the diagonal entry A[i, i] = 1.
#   2. For the right-hand side (RHS) vector L:
#      - Set L[i] = boundary value for all DOFs in that group.
#
# This ensures that when solving the system, the unknown at those DOFs is
# exactly equal to the prescribed boundary value.
#
# Example:
#   Suppose we have a 3×3 system (DOFs 0–8 arranged in a 3×3 grid) and we
#   want to enforce:
#       - u = 1 on the bottom boundary [0, 1, 2]
#       - u = 2 on the right boundary  [2, 5, 8]
#
#   Call:
#       A_bc, L_bc = dirichlet_bc(A, L, ([0, 1, 2], [2, 5, 8]), [1.0, 2.0])
#
#   The returned system will guarantee:
#       u(0)=1, u(1)=1, u(2)=2, u(5)=2, u(8)=2
# -----------------------------------------------------------------------------

import numpy as np
from typing import Tuple


def dirichlet_bc(
    a: np.ndarray, l: np.ndarray, dof_groups: tuple, bc_values: list[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Dirichlet boundary conditions to the global system [A]{u} = {L}.

    Parameters
    ----------
    a : np.ndarray
        Global stiffness (or system) matrix of shape (N, N).
    l : np.ndarray
        Global right-hand side (load/source) vector of shape (N,).
    dof_groups : tuple of lists
        Each entry is a list of DOF indices belonging to one boundary.
        Example: ([0, 1, 2], [2, 5, 8])
    bc_values : list of float
        List of boundary values corresponding to each dof_group.
        Must be the same length as dof_groups.

    Returns
    -------
    A : np.ndarray
        Modified system matrix with Dirichlet BCs applied.
    L : np.ndarray
        Modified RHS vector with Dirichlet BCs applied.
    """

    # Loop over all boundary groups
    for index_list in dof_groups:
        # Zero out rows corresponding to constrained DOFs
        for r in index_list:
            if 0 <= r < a.shape[0]:
                a[r] = [0] * a.shape[1]

        # Set diagonal entries to 1 (enforcing identity rows)
        for r in index_list:
            if 0 <= r < a.shape[0] and 0 <= r < a.shape[1]:
                a[r][r] = 1

    # Modify the RHS vector for each group
    for index_list, val in zip(dof_groups, bc_values):
        for r in index_list:
            l[r] = val  # enforce prescribed boundary value

    return a, l
