"""
Definition of matrices for a global system of equations used in 1D finite element methods.

Intervals:
- Interpolation in an arbitrary interval

Basis function types:
- LinearBasis
- QuadraticBasis
- HermiteBasis
"""

import numpy as np
from numpy import ndarray


def assemble_global_matrices(
    mesh_nodes: ndarray,
    basis_function,
    element_matrix_function: callable,
    element_vector_function: callable,
    nb_quadrature_points: int = 3,
):
    """
    Assemble global stiffness matrix and load vector for 1D finite element method.

    Parameters
    ----------
    mesh_nodes : np.ndarray
        Global mesh node positions.
    basis_function : BasisFunction
        Basis function class (Linear, Quadrature, and Hermite basis functions).
    element_matrix_function : callable
        Function to compute element matrix, e.g., stiffness_matrix_fourth_derivative.
    element_vector_function : callable
        Function to compute element load vector, e.g., load_vector.
    nb_quadrature_points : int
        Number of Gauss quadrature points.

    Returns
    -------
    k_global : np.ndarray
        Assembled global stiffness and mass matrix.
    f_global : np.ndarray
        Assembled global load vector.
    """
    nb_elements = len(mesh_nodes) - 1
    nb_nodes = basis_function.nb_nodes
    nb_dofs_per_node = basis_function.dof_per_node

    total_dofs = nb_elements * (nb_nodes - nb_dofs_per_node) + nb_dofs_per_node

    # Initialize global matrix and vector
    global_matrix = np.zeros((total_dofs, total_dofs))
    global_vector = np.zeros(total_dofs)

    for e in range(nb_elements):
        y_n = mesh_nodes[e : e + 2]

        # Compute element matrix and load vector
        element_matrix = element_matrix_function(
            nb_quadrature_points, y_n, basis_function
        )
        element_vector = element_vector_function(
            nb_quadrature_points, y_n, basis_function
        )

        # Compute the position of local element matrix and load vector to global
        start = e * (nb_nodes - nb_dofs_per_node)
        end = start + nb_nodes

        # Assemble local element matrix and load vector to global
        global_matrix[start:end, start:end] += element_matrix
        global_vector[start:end] += element_vector
    # Remove numerical noise
    tol = 1e-14
    global_matrix[np.abs(global_matrix) < tol] = 0.0
    global_vector[np.abs(global_vector) < tol] = 0.0

    return global_matrix, global_vector
