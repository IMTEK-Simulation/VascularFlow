"""
Definition of matrices and vectors for a global system of equations used in 1D and 2D finite element methods.

Intervals:
- 1D:Interpolation in an arbitrary interval
- 2D:Square reference element Ê = [−1, 1]×[−1, 1] centered at the origin of the Cartesian (ξ , η) coordinate system

Basis function types:
1D:
    - LinearBasis
    - QuadraticBasis
    - HermiteBasis
2D:
    - Bi-linear shape functions
    - adini clough melosh (ACM)
"""

import numpy as np
from numpy import ndarray
import scipy.sparse as scysparse

from VascularFlow.Numerics.BasisFunctions import (
    BasisFunction,
    ACMShapeFunctions,
)
from VascularFlow.Numerics.ElementMatrices import (
    element_matrices_or_vectors_2d,
    element_matrices_or_vectors_acm_2d,
)
from VascularFlow.Numerics.Connectivity2D import (
    build_connectivity,
    build_connectivity_acm,
)


def assemble_global_matrices(
    mesh_nodes: ndarray,
    basis_function: BasisFunction,
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


########################## Assembly of matrices and vectors used in 2D finite element methods. #########################


def assemble_global_matrices_vectors_2d(
    shape_function,
    domain_length: float,
    domain_height: float,
    n_x: int,
    nb_quad_pts_2d: int = 9,
    source_func: callable = None,
):
    """
    Assemble the global stiffness matrix K, global mass matrix M,
    and global source vector F for a structured 2D rectangular mesh
    of four-node bilinear (Q1) elements.

    Parameters
    ----------
    shape_function : ShapeFunction
        Instance of a shape function class implementing `.eval()` and `.first_derivative()`.
    domain_length : float
        Length of the rectangular domain in the x-direction.
    domain_height : float
        Height of the rectangular domain in the y-direction.
    n_x : int
        Number of nodes in the horizontal (x) direction.
    nb_quad_pts_2d : int, optional
        Number of 2D Gaussian quadrature points (default: 9).
    source_func : callable, optional
        Source term function f(ξ, η), takes two arguments (local coordinates).
        If None, the source vector will be zero.

    Returns
    -------
    K : np.ndarray
        Global stiffness matrix of shape (N_nodes, N_nodes).
    M : np.ndarray
        Global mass matrix of shape (N_nodes, N_nodes).
    F : np.ndarray
        Global source vector of shape (N_nodes,).
    elements : list[list[int]]
        Element connectivity list, where each sub-list contains
        the 4 global node indices for that element.
    """

    # Element width in x-direction
    dx = domain_length / (n_x - 1)

    # Number of nodes in y-direction (for a structured 2D rectangular mesh dx = dy)
    n_y = int(domain_height / dx) + 1

    # Build element connectivity
    elements, N_nodes = build_connectivity(n_x, n_y, one_based=False)

    # For uniform mesh with constant coefficients, local matrices are identical → compute once
    K_e = element_matrices_or_vectors_2d(
        shape_function, nb_quad_pts_2d, dx, kind="stiffness"
    )
    M_e = element_matrices_or_vectors_2d(
        shape_function, nb_quad_pts_2d, dx, kind="mass"
    )
    F_e = element_matrices_or_vectors_2d(
        shape_function, nb_quad_pts_2d, dx, kind="source", f=source_func
    )

    # Initialize global matrices and vector
    K = np.zeros((N_nodes, N_nodes))
    M = np.zeros((N_nodes, N_nodes))
    F = np.zeros(N_nodes)

    # Assembly process
    for conn in elements:  # Loop over all elements
        # conn: [bottom-left, bottom-right, top-right, top-left] (0-based global indices)
        for a, I in enumerate(conn):  # Local row index a maps to global row I
            F[I] += F_e[a]  # Add local source term to global vector
            for b, J in enumerate(conn):  # Local col index b maps to global col J
                K[I, J] += K_e[a, b]
                M[I, J] += M_e[a, b]

    return K, M, F, n_y


def assemble_global_matrices_vectors_2d_acm(
    shape_function,
    domain_length: float,
    domain_height: float,
    n_x: int,
    n_y: int,
    nb_quad_pts_2d: int = 9,
):
    """
    Assemble the global stiffness matrix K, and global source vector F
    for a structured 2D rectangular mesh
    of four-node acm elements.

    Parameters
    ----------
    shape_function : ShapeFunction
        Instance of a shape function class implementing `.eval()` and `.second_derivative()`.
    domain_length : float
        Length of the rectangular domain in the x-direction.
    domain_height : float
        Height of the rectangular domain in the y-direction.
    n_x : int
        Number of nodes in the horizontal (x) direction.
    n_y : int
        Number of nodes in the vertical (y) direction.
    nb_quad_pts_2d : int, optional
        Number of 2D Gaussian quadrature points (default: 9).

    Returns
    -------
    K : np.ndarray
        Global stiffness matrix of shape (N_dofs, N_dofs).
    F : np.ndarray
        Global source vector of shape (N_dofs,).
    """

    # Element width in x-direction
    dx = domain_length / (n_x - 1)

    # Element width in x-direction
    dy = domain_height / (n_y - 1)

    # Build element connectivity
    elements_dof, N_nodes, N_dofs = build_connectivity_acm(n_x, n_y, one_based=False)

    # For uniform mesh with constant coefficients, local matrices are identical → compute once
    K_e = element_matrices_or_vectors_acm_2d(
        shape_function, nb_quad_pts_2d, dx, dy
    )[0]
    F_e = element_matrices_or_vectors_acm_2d(
        shape_function, nb_quad_pts_2d, dx, dy
    )[1]

    # Initialize global matrices and vector
    #K = np.zeros((N_dofs, N_dofs))
    K = scysparse.lil_matrix((N_dofs, N_dofs), dtype=np.float64)
    F = np.zeros(N_dofs)

    # Assembly process
    for conn in elements_dof:  # Loop over all elements
        # conn: [bottom-left, bottom-right, top-right, top-left] (0-based global indices)
        for a, I in enumerate(conn):  # Local row index a maps to global row I
            F[I] += F_e[a]  # Add local source term to global vector
            for b, J in enumerate(conn):  # Local col index b maps to global col J
                K[I, J] += K_e[a, b]

    return K, F
