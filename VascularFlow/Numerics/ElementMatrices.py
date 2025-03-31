"""
Definition of matrices for a single element used in 1D finite element methods.

Intervals:
- Interpolation in an arbitrary interval

Basis function types:
- LinearBasis
- QuadraticBasis
- HermiteBasis
"""

import numpy as np

from VascularFlow.Numerics.BasisFunctions import BasisFunction
from VascularFlow.Numerics.Quadrature import gaussian_quadrature


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
