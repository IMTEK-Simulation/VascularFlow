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

from VascularFlow.Numerics.BasisFunctions import (
    BasisFunction,
    ACMShapeFunctions,
    BilinearShapeFunctions,
)
from VascularFlow.Numerics.ElementMatrices import (
    element_matrices_or_vectors_2d,
    element_matrices_or_vectors_acm_2d,
)
from VascularFlow.Numerics.Connectivity2D import (
    build_connectivity,
    build_connectivity_dofs,
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


def global_matrices_vectors_2d_acm(
    shape_function,
    l: float,
    h: float,
    n_x: int,
    nb_quad_pts_2d: int = 9,
    source_func: callable = None,
):
    """
    Assemble the global stiffness matrix K
    and global source vector F for a structured 2D rectangular mesh
    of four-node ACM (Q1) elements.

    Parameters
    ----------
    shape_function : ShapeFunction
        Instance of a shape function class implementing `.eval()` and `.first_derivative()`.
    l : float
        Length of the rectangular domain in the x-direction.
    h : float
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
    Kxx : np.ndarray
        Global stiffness matrix (x-direction) of shape (N_nodes, N_nodes).
    Kyy : np.ndarray
        Global stiffness matrix (y-direction) of shape (N_nodes, N_nodes).
    Kxy : np.ndarray
        Global stiffness matrix (xy-direction) of shape (N_nodes, N_nodes).
    F : np.ndarray
        Global source vector of shape (N_nodes,).
    elements : list[list[int]]
        Element connectivity list, where each sub-list contains
        the 4 global node indices for that element.
    """

    # Element width in x-direction
    dx = l / (n_x - 1)

    # Number of nodes in y-direction (for a structured 2D rectangular mesh dx = dy)
    # n_y = int(h / dx) + 1
    n_y = n_x  # For square elements, n_y = n_x
    # Build element connectivity
    dofs_per_node = 3  # For ACM shape functions, each node has 3 degrees of freedom
    elements, N_nodes = build_connectivity(n_x, n_y, one_based=False)
    element_dofs, N_nodes, N_dofs = build_connectivity_dofs(
        n_x, n_y, dofs_per_node=dofs_per_node, one_based=False
    )

    # For uniform mesh with constant coefficients, local matrices are identical → compute once
    K_xx, K_yy, K_xy, F_e = element_matrices_or_vectors_acm_2d(
        shape_function, nb_quad_pts_2d, dx, source_func=source_func
    )

    # Initialize global matrices and vector
    Kxx = np.zeros((N_dofs, N_dofs))
    Kyy = np.zeros((N_dofs, N_dofs))
    Kxy = np.zeros((N_dofs, N_dofs))
    F = np.zeros(N_dofs)

    # Assembly process
    for conn in element_dofs:  # Loop over all elements
        # print(f"conn: {conn}")  # Debugging line to check connectivity
        # conn: [bottom-left, bottom-right, top-right, top-left] (0-based global indices)
        for a, I in enumerate(conn):  # Local row index a maps to global row I
            # print(f"Local row index a: {a}, Global row index I: {I}")  # Debugging line to check indices
            F[I] += F_e[a]  # Add local source term to global vector
            for b, J in enumerate(conn):  # Local col index b maps to global col J
                # print(f"Local col index b: {b}, Global col index J: {J}")  # Debugging line to check indices
                Kxx[I, J] += K_xx[a, b]
                Kyy[I, J] += K_yy[a, b]
                Kxy[I, J] += K_xy[a, b]

    return Kxx, Kyy, Kxy, F, elements


if __name__ == "__main__":
    # Example usage
    n_x = 20  # Number of nodes in x-direction
    nx, ny = 20, 20  # 3 Knoten in x, 3 in y
    w, l = 200 * 10**-6, 1000 * 10**-6  # Width and length of the plate
    # w, l = 10.0, 50.0 # Width and length of the plate
    h = 20 * 10**-6  # Thickness of the plate
    E = 2 * 10**6
    nu = 0.3
    q = 1.5 * 10**-9
    v = q / (h * w)
    phi = 1000
    D = 2 * E * h**3 / (3 * (1 - nu**2))
    kxx_global, kyy_global, kxy_global, f_global, elements = (
        global_matrices_vectors_2d_acm(
            shape_function=ACMShapeFunctions(),
            l=l,
            h=w,
            n_x=n_x,
            nb_quad_pts_2d=9,
            source_func=lambda x, y: 1000e11
        )
    )
    k_total = kxx_global + kyy_global + kxy_global

    for i in range(3 * n_x**2):
        # bottom wall
        if i < 3 * n_x:
            k_total[i, :] = 0.0
            f_global[i] = 0.0
            k_total[i, i] = 1.0
        # left wall
        #if i % (3 * n_x) == 0:
        #    k_total[i, :] = 0.0
        #    f_global[i] = 0.0
        #    k_total[i, i] = 1.0
        #if i % ((3 * n_x)) == 1:
        #    k_total[i, :] = 0.0
        #    f_global[i] = 0.0
        #    k_total[i, i] = 1.0
        #if i % ((3 * n_x)) == 2:
        #    k_total[i, :] = 0.0
        #    f_global[i] = 0.0
        #    k_total[i, i] = 1.0
        # right wall
        #if i % (3 * n_x) == 3 * n_x - 1:
        #    k_total[i, :] = 0.0
        #    f_global[i] = 0.0
        #    k_total[i, i] = 1.0
        #if i % (3 * n_x) == 3 * n_x - 2:
        #    k_total[i, :] = 0.0
        #    f_global[i] = 0.0
        #    k_total[i, i] = 1.0
        #if i % (3 * n_x) == 3 * n_x - 3:
        #    k_total[i, :] = 0.0
        #    f_global[i] = 0.0
        #    k_total[i, i] = 1.0
        # top wall
        if i > 3 * n_x * (n_x - 1):
            k_total[i, :] = 0.0
            f_global[i] = 0.0
            k_total[i, i] = 1.0
    sol = np.linalg.solve(k_total, f_global)
    #w_tilde = sol
    #sol = w_tilde/h

import numpy as np
import matplotlib.pyplot as plt

def plot_fem_solution_2d(sol, nx, ny, w, l, dof=0, levels=50, title="FEM solution (2D)"):

    Z = sol[dof::3].reshape(ny, nx)

    # coordinates
    x = np.linspace(0, w, nx)
    y = np.linspace(0, l, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # plot
    plt.figure(figsize=(6.5, 5.5))
    cf = plt.contourf(X, Y, Z, levels=levels, cmap="viridis")
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.title(title)
    plt.gca().set_aspect("equal")
    cbar = plt.colorbar(cf)
    cbar.set_label("Field value")
    plt.tight_layout()
    plt.show()

# Example call (plots the first DOF per node):
plot_fem_solution_2d(sol, nx, ny, w, l, dof=0, levels=60)




