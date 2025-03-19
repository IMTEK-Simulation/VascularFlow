import numpy as np

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


def first_first(nb_quad_pts: int, basis_function: BasisFunction):
    return element_matrix(
        nb_quad_pts, basis_function.first_derivative, basis_function.first_derivative
    )


def second_second(nb_quad_pts: int, basis_function: BasisFunction):
    return element_matrix(
        nb_quad_pts, basis_function.first_derivative, basis_function.first_derivative
    )
