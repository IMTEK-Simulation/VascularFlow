import numpy as np
import pytest

from VascularFlow.Numerics.Quadrature import gaussian_quadrature1, gaussian_quadrature

_functions = [
    (lambda x: x, 0, 1),
    (lambda x: x**2, 2 / 3, 2),
    (lambda x: x**3, 0, 2),
    (lambda x: x**4, 2 / 5, 3),
]


@pytest.mark.parametrize("function", _functions)
@pytest.mark.parametrize("nb_quad_pts", range(1, 6))
def test_gaussian_quadrature1(function, nb_quad_pts, atol=1e-6):
    f, analytic_integral, min_nb_quad_pts = function
    if nb_quad_pts < min_nb_quad_pts:
        # This test cannot work, because the number of quadrature points is too low
        return
    numerical_integral = gaussian_quadrature1(nb_quad_pts, f)
    np.testing.assert_allclose(numerical_integral, analytic_integral, atol=atol)


def test_gaussian_quadrature():
    def f(x):
        return x + x**3

    np.testing.assert_allclose(gaussian_quadrature(2, 1, 3, f), 24.0)
