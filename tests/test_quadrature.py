import numpy as np
import pytest

from VascularFlow.Numerics.BasisFunctions import HermiteBasis
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


@pytest.mark.parametrize("nb_quad_pts", [3])  # Different quadrature points
def test_integral_first_hermite_basis(nb_quad_pts):
    basis_function = HermiteBasis()  # Create instance
    func = lambda x: basis_function.eval(x)[3]  # First element of eval(x)

    integral_value = gaussian_quadrature(nb_quad_pts, 0, 1, func)  # Integrate over [0,1]

    expected_value = -0.08  # Expected integral (analytically computed for reference)

    assert np.isclose(integral_value, expected_value, atol=1e-2), \
        f"Integral value is {integral_value}, expected {expected_value}"

    print(f"Integral of first HermiteBasis function over [0,1] ≈ {integral_value}")

