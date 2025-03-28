import numpy as np
from scipy.sparse import diags

from VascularFlow.Numerics.BasisFunctions import BasisFunction
from VascularFlow.Numerics.Quadrature import gaussian_quadrature


def element_matrix(nb_quad_pts: int, f: callable, g: callable):
    def outer_product(x_q: np.array) -> np.array:
        f_nq = f(x_q)
        g_nq = g(x_q)
        n, q = f_nq.shape
        assert g_nq.shape == (n, q)
        return f_nq.reshape(n, 1, q) * g_nq.reshape(1, n, q)

    return gaussian_quadrature(nb_quad_pts, 0, 1, outer_product)


def stiffness_matrix_fourth_derivative(
    nb_quad_pts: int, dx_e: float, basis_function: BasisFunction
):
    """Stiffness matrix used in 1D Euler–Bernoulli beam equation (steady) for displacement calculation."""
    dx_e_stiffness_matrix = [
        [1 / dx_e**3, 1 / dx_e**2, 1 / dx_e**3, 1 / dx_e**2],
        [1 / dx_e**2, 1 / dx_e, 1 / dx_e**2, 1 / dx_e],
        [1 / dx_e**3, 1 / dx_e**2, 1 / dx_e**3, 1 / dx_e**2],
        [1 / dx_e**2, 1 / dx_e, 1 / dx_e**2, 1 / dx_e],
    ]
    return (
        element_matrix(
            nb_quad_pts,
            basis_function.second_derivative,
            basis_function.second_derivative,
        )
        * dx_e_stiffness_matrix
    )


def mass_matrix_fourth_derivatives(
    nb_quad_pts: int, dx_e: float, basis_function: BasisFunction
):
    """mass matrix used in 1D Euler–Bernoulli beam equation (transient) for displacement calculation."""
    dx_e_mass_matrix = [
        [dx_e, dx_e**2, dx_e, dx_e**2],
        [dx_e**2, dx_e**3, dx_e**2, dx_e**3],
        [dx_e, dx_e**2, dx_e, dx_e**2],
        [dx_e**2, dx_e**3, dx_e**2, dx_e**3],
    ]
    return (
        element_matrix(nb_quad_pts, basis_function.eval, basis_function.eval)
        * dx_e_mass_matrix
    )


def load_vector_fourth_derivatives(dx_e: float):
    """load vector used in 1D Euler–Bernoulli beam equation for displacement calculation"""
    return np.array([dx_e / 2, dx_e**2 / 12, dx_e / 2, -(dx_e**2) / 12])


def stiffness_matrix_first_derivative(nb_quad_pts: int, basis_function: BasisFunction):
    """stiffness matrix used in Navier-Stocks equations for fluid pressure and flow rate calculation"""
    return element_matrix(
        nb_quad_pts, basis_function.eval, basis_function.first_derivative
    )


def load_vector_first_derivative(dx_e: float):
    """load vector used in Navier-Stocks equations for fluid pressure and flow rate calculation"""
    return np.array([dx_e / 2, dx_e / 2])


def stiffness_matrix_second_derivative(nb_quad_pts: int, basis_function: BasisFunction):
    return element_matrix(
        nb_quad_pts, basis_function.first_derivative, basis_function.first_derivative
    )
