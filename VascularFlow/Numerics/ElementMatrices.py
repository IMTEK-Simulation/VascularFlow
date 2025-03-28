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

def eval_first(nb_quad_pts: int, basis_function: BasisFunction):
    # Stiffness matrix used for pressure and flow rate calculation in Navier-Stokes equations
    return element_matrix(
        nb_quad_pts, basis_function.eval, basis_function.first_derivative
    )

def first_first(nb_quad_pts: int, basis_function: BasisFunction):
    return element_matrix(
        nb_quad_pts, basis_function.first_derivative, basis_function.first_derivative
    )

def second_second(nb_quad_pts: int, dx: float, basis_function: BasisFunction):
    # Stiffness matrix used for displacement calculation in Euler–Bernoulli beam equation
    dx_matrix = [
        [1 / dx**3, 1 / dx**2, 1 / dx**3, 1 / dx**2],
        [1 / dx**2, 1 / dx, 1 / dx**2, 1 / dx],
        [1 / dx**3, 1 / dx**2, 1 / dx**3, 1 / dx**2],
        [1 / dx**2, 1 / dx, 1 / dx**2, 1 / dx],
    ]

    return element_matrix(
        nb_quad_pts, basis_function.second_derivative, basis_function.second_derivative
    ) * dx_matrix











def dx_matrix_mass(dx):
    return [
        [dx, dx**2, dx, dx**2],
        [dx**2, dx**3, dx**2, dx**3],
        [dx, dx**2, dx, dx**2],
        [dx**2, dx**3, dx**2, dx**3],
    ]











def force_matrix(dx: int):
    return np.array([dx / 2, dx**2 / 12, dx / 2, -(dx**2) / 12])


def force_matrix_pressure(dx: int):
    return np.array([dx / 2, dx / 2])


def mass_matrix(nb_quad_pts: int, dx: int, basis_function: BasisFunction):
    return element_matrix(
        nb_quad_pts, basis_function.eval, basis_function.eval
    ) * dx_matrix_mass(dx)
