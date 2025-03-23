import numpy as np
import pytest

from VascularFlow.Numerics.BasisFunctions import LinearBasis, QuadraticBasis, \
    HermiteBasis, CubicBasis


@pytest.mark.skip
def test_plot_basis_function(basis_function=HermiteBasis()):
    import matplotlib.pyplot as plt
    x = np.linspace(0, 1, 101)
    vs = basis_function.eval(x)
    for v in vs:
        plt.plot(x, v, '-')
    plt.show()


@pytest.mark.parametrize('basis_function_class',
                         [LinearBasis, QuadraticBasis, CubicBasis, HermiteBasis])
def test_first_derivative(basis_function_class, delta=1e-6, rtol=1e-6):
    basis_function = basis_function_class()
    x_i = np.array([0.1, 0.3, 0.7, 0.9])
    v_ni = basis_function.eval(x_i)
    d_ni = basis_function.first_derivative(x_i)
    assert v_ni.shape == (basis_function.nb_nodes, len(x_i))
    assert v_ni.shape == d_ni.shape

    # Finite-differences test of the derivative
    v_plus_ni = basis_function.eval(x_i + delta)
    v_minus_ni = basis_function.eval(x_i - delta)
    d_fd_ni = (v_plus_ni - v_minus_ni) / (2 * delta)
    np.testing.assert_allclose(d_ni, d_fd_ni, rtol=rtol)


@pytest.mark.parametrize('basis_function_class',
                         [QuadraticBasis, CubicBasis, HermiteBasis])
def test_second_derivative(basis_function_class, delta=1e-6, rtol=1e-6):
    basis_function = basis_function_class()
    x_i = np.array([0.1, 0.3, 0.7, 0.9])
    v_ni = basis_function.first_derivative(x_i)
    d_ni = basis_function.second_derivative(x_i)
    assert v_ni.shape == (basis_function.nb_nodes, len(x_i))
    assert v_ni.shape == d_ni.shape

    # Finite-differences test of the derivative
    v_plus_ni = basis_function.first_derivative(x_i + delta)
    v_minus_ni = basis_function.first_derivative(x_i - delta)
    d_fd_ni = (v_plus_ni - v_minus_ni) / (2 * delta)
    np.testing.assert_allclose(d_ni, d_fd_ni, rtol=rtol)
