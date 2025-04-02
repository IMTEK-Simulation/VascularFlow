"""
Test the first derivative of an arbitrary functions using finite difference method.
Compare the finite difference method with analytical solution.

Functions:
- f(x) = 2 * x
- f(x) = x²
- f(x) = sin(x)
- f(x) = e^x

"""

from VascularFlow.Numerics.ArrayFirstDerivative import array_first_derivative

import numpy as np
import pytest


@pytest.mark.parametrize(
    "func, d_func",
    [
        (lambda x: 2 * x, lambda x: 2 * np.ones_like(x)),
        (lambda x: x**2, lambda x: 2 * x),
        (np.sin, np.cos),
        (np.exp, np.exp),
    ],
)
def test_array_first_derivative(func, d_func):
    x = np.linspace(0, 2 * np.pi, 100)
    f = func(x)
    expected_df = d_func(x)

    computed_df = array_first_derivative(f, x)

    # Allowing for numerical error, especially at boundaries
    np.testing.assert_allclose(
        computed_df[1:-1], expected_df[1:-1], rtol=1e-2, atol=1e-3
    )
