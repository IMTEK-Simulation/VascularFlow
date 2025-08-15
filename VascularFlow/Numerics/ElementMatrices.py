"""
Definition of matrices for a single element used in 1D finite element methods.
Definition of matrices for a single reference quadrilateral element used in 2D finite element methods.

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

from VascularFlow.Numerics.BasisFunctions import BasisFunction
from VascularFlow.Numerics.Quadrature import gaussian_quadrature, integrate_over_square

#################### Definition of matrices for a single element used in 1D finite element methods. ####################


def element_matrix(nb_quad_pts: int, y_n: np.ndarray, f: callable, g: callable):
    """
    compute ∫f.g dx using Gaussian quadrature rule in an arbitrary interval used in definition of element matrices.

    Parameters
    ----------
    nb_quad_pts : int
        Number of quadrature points.
    y_n : np.ndarray
        The integral arbitrary interval.
    f : callable
        An arbitrary function.
    g : callable
        An arbitrary function.

    Returns
    -------
    gaussian_quadrature
        The value of the Gaussian quadrature rule.
    """

    def outer_product(x_q: np.array) -> np.array:
        f_nq = f(x_q)
        g_nq = g(x_q)
        n, q = f_nq.shape
        assert g_nq.shape == (n, q)
        return f_nq.reshape(n, 1, q) * g_nq.reshape(1, n, q)

    return gaussian_quadrature(nb_quad_pts, y_n[0], y_n[1], outer_product)


def stiffness_matrix_fourth_derivative(
    nb_quad_pts: int, y_n: np.ndarray, basis_function: BasisFunction
):
    """Return the stiffness matrix used in Euler–Bernoulli beam equation."""

    nb_nodes = basis_function.nb_nodes

    def f(x_q):
        return np.stack(
            [
                basis_function.interpolate_second_derivative(
                    y_n, np.eye(nb_nodes)[i], x_q
                )
                for i in range(nb_nodes)
            ]
        )

    return element_matrix(nb_quad_pts, y_n, f, f)


def mass_matrix_fourth_derivatives(
    nb_quad_pts: int, y_n: np.ndarray, basis_function: BasisFunction
):
    """Return the mass matrix used in Euler–Bernoulli beam equation."""
    nb_nodes = basis_function.nb_nodes

    def f(x_q):
        return np.stack(
            [
                basis_function.interpolate(y_n, np.eye(nb_nodes)[i], x_q)
                for i in range(nb_nodes)
            ]
        )

    return element_matrix(nb_quad_pts, y_n, f, f)


def load_vector(
    nb_quad_pts: int,
    y_n: np.ndarray,
    basis_function: BasisFunction,
    g: callable = lambda x: np.ones_like(x),
):
    """Return the load vector (The external force matrix) used in Euler–Bernoulli beam equation and Navier-Stocks equations."""
    nb_nodes = basis_function.nb_nodes

    def f(x_q):
        return np.stack(
            [
                basis_function.interpolate(y_n, np.eye(nb_nodes)[i], x_q)
                for i in range(nb_nodes)
            ]
        )

    def g_vec(x_q):
        return np.ones_like(x_q) * g(x_q)

    return gaussian_quadrature(
        nb_quad_pts, y_n[0], y_n[1], lambda x_q: f(x_q) * g_vec(x_q)
    )


def stiffness_matrix_first_derivative(
    nb_quad_pts: int,
    y_n: np.ndarray,
    basis_function: BasisFunction,
):
    """Return the stiffness matrix used in Navier-Stocks equation."""
    nb_nodes = basis_function.nb_nodes

    def f(x_q):
        return np.stack(
            [
                basis_function.interpolate(y_n, np.eye(nb_nodes)[i], x_q)
                for i in range(nb_nodes)
            ]
        )

    def g(x_q):
        return np.stack(
            [
                basis_function.interpolate_first_derivative(
                    y_n, np.eye(nb_nodes)[j], x_q
                )
                for j in range(nb_nodes)
            ]
        )

    return element_matrix(nb_quad_pts, y_n, f, g)


def stiffness_matrix_second_derivative(
    nb_quad_pts: int, y_n: np.ndarray, basis_function: BasisFunction
):
    nb_nodes = basis_function.nb_nodes

    def f(x_q):
        return np.stack(
            [
                basis_function.interpolate_first_derivative(
                    y_n, np.eye(nb_nodes)[i], x_q
                )
                for i in range(nb_nodes)
            ]
        )

    return element_matrix(nb_quad_pts, y_n, f, f)


######## Definition of matrices for a single reference quadrilateral element used in 2D finite element methods. ########


def element_matrices_or_vectors_2d(
    shape_func, nb_quad_pts_2d: int, dx: float, kind="mass", f=None
) -> np.ndarray:
    """
    Compute the element matrix (mass, stiffness) or load vector (source term)
    for a given shape function on the reference square [-1, 1] × [-1, 1]
    using 2D Gaussian quadrature.

    Parameters
    ----------
    shape_func : ShapeFunction
        Shape function instance with `.eval(s, n)` and `.first_derivative(s, n)`.
    nb_quad_pts_2d : int
        Number of quadrature points in 2D (4 or 9 supported).
    dx : float
        Element side length (square elements assumed).
    kind : {"mass", "stiffness", "source"}
        Type of computation:
            - "mass"     : M_ij = ∬ φ_i φ_j |J| dξ dη
            - "stiffness": K_ij = ∬ (∇φ_i ⋅ ∇φ_j) |J| dξ dη
            - "source"   : F_i  = ∬ φ_i f(ξ,η) |J| dξ dη
    f : callable, optional
        Source term function f(ξ, η) for "source" case.

    Returns
    -------
    np.ndarray
        Element matrix (nb_nodes × nb_nodes) or vector (nb_nodes,) depending on `kind`.
    """
    nb_nodes = shape_func.nb_nodes

    # Jacobin for a square element mapping from reference coords to physical coords
    J = np.array([[dx / 2, 0.0], [0.0, dx / 2]])
    detJ = np.linalg.det(J)  # Area scaling factor

    if kind == "mass":
        M = np.zeros((nb_nodes, nb_nodes))
        for i in range(nb_nodes):
            for j in range(nb_nodes):
                # integrand in reference coords; multiply by |detJ| to map to physical element
                integrand = (
                    lambda s, n: shape_func.eval(s, n)[i]
                    * shape_func.eval(s, n)[j]
                    * detJ
                )
                M[i, j] = integrate_over_square(integrand, nb_quad_pts_2d)
        return M

    elif kind == "stiffness":
        # Ensure the mapping is valid
        if detJ <= 0:
            raise ValueError("Invalid element mapping: |det(J)| must be > 0.")

        # Inverse-transpose of J for transforming gradients
        J_inv_T = np.linalg.inv(J).T

        K = np.zeros((nb_nodes, nb_nodes))
        for i in range(nb_nodes):
            for j in range(nb_nodes):

                def integrand(s, n):
                    # grad in reference coords: shape (nb_nodes, 2)
                    grad_hat = shape_func.first_derivative(s, n)
                    # map to physical coords: ∇φ = ∇_ref φ · J^{-T}
                    grad_phi = grad_hat @ J_inv_T  # shape (nb_nodes, 2)
                    return (grad_phi[i] @ grad_phi[j]) * detJ

                K[i, j] = integrate_over_square(integrand, nb_quad_pts_2d)
        return K

    elif kind == "source":
        if f is None:
            raise ValueError("Source function f must be provided for kind='source'.")
        F = np.zeros(nb_nodes)
        for i in range(nb_nodes):
            integrand = lambda s, n: shape_func.eval(s, n)[i] * f(s, n) * detJ
            F[i] = integrate_over_square(integrand, nb_quad_pts_2d)
        return F

    else:
        raise ValueError("kind must be 'mass' or 'stiffness'")
