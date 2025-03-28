import numpy as np
import pytest

from VascularFlow.Numerics.BasisFunctions import (
    LinearBasis,
    QuadraticBasis,
    HermiteBasis,
)


@pytest.mark.parametrize("basis_function_class", [HermiteBasis])
def test_basis_functions(basis_function_class, delta=1e-6, rtol=1e-6, plot=True):
    basis_function = basis_function_class()
    x_i = np.array([0.1, 0.3, 0.7, 0.9])
    v_ni = basis_function.eval(x_i)
    d_ni = basis_function.first_derivative(x_i)
    sd_ni = basis_function.second_derivative(x_i)
    assert v_ni.shape == (basis_function.nb_nodes, len(x_i))
    assert v_ni.shape == d_ni.shape
    assert d_ni.shape == sd_ni.shape
    if plot:
        import matplotlib.pyplot as plt

        x = np.linspace(0, 1, 10)
        fig, ax = plt.subplots(3, 1)
        for i in range(basis_function.nb_nodes):
            ax[0].plot(x, basis_function.eval(x)[i], label=f"Basis {i}")
            ax[1].plot(x, basis_function.first_derivative(x)[i], label=f"Basis {i}")
            ax[2].plot(x, basis_function.second_derivative(x)[i], label=f"Basis {i}")
        ax[0].set_title("basis function")
        ax[1].set_title("first derivative")
        ax[2].set_title("second derivative")

        for a in ax:
            a.legend()
            a.grid(True)

        plt.tight_layout()
        plt.show()

    # Finite-differences test of the derivative
    v_plus_ni = basis_function.eval(x_i + delta)
    v_minus_ni = basis_function.eval(x_i - delta)
    d_fd_ni = (v_plus_ni - v_minus_ni) / (2 * delta)
    np.testing.assert_allclose(d_ni, d_fd_ni, rtol=rtol)

    # Finite-difference test of the derivative
    v_plus_ni = basis_function.eval(x_i + delta)
    v_ni = basis_function.eval(x_i)
    v_minus_ni = basis_function.eval(x_i - delta)
    ds_fd_ni = (v_plus_ni - 2 * v_ni + v_minus_ni) / (delta**2)
    np.testing.assert_allclose(sd_ni, ds_fd_ni, rtol=1e-03)
