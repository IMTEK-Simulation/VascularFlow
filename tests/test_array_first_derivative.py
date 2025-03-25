from VascularFlow.Numerics.ArrayFirstDerivative import array_first_derivative

import numpy as np
import pytest

@pytest.mark.parametrize(
    "func, dfunc, description",
    [
        (lambda x: 2 * x, lambda x: 2 * np.ones_like(x), "linear"),
        (lambda x: x**2, lambda x: 2 * x, "quadratic"),
        (np.sin, np.cos, "sine"),
        (np.exp, np.exp, "exponential"),
    ]
)
def test_array_first_derivative(func, dfunc, description):
    x = np.linspace(0, 2 * np.pi, 100)
    dx = x[1] - x[0]
    f = func(x)
    expected_df = dfunc(x)

    computed_df = array_first_derivative(f, dx)

    # Allowing for numerical error, especially at boundaries
    np.testing.assert_allclose(computed_df[1:-1], expected_df[1:-1], rtol=1e-2, atol=1e-3, err_msg=f"Failed on {description}")
