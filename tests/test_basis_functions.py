"""
Tests for evaluating the correctness of basis functions and shape functions used in finite element methods.

This module verifies:
- Shape consistency of evaluations and derivatives (for 1D basis functions).
- Accuracy of first and second derivatives using finite-difference approximation (for 1D basis functions).
- Optional visualization of basis functions and their derivatives (for 1D basis functions and 2D shape functions).

Tested basis types:
- LinearBasis
- QuadraticBasis
- HermiteBasis
- Bi-linear shape functions
- Adini-Clough-Melosh (ACM)
"""

import numpy as np
import pytest

from VascularFlow.Numerics.BasisFunctions import (
    LinearBasis,
    QuadraticBasis,
    HermiteBasis,
    BilinearShapeFunctions,
)


@pytest.mark.parametrize(
    "basis_function_class", [LinearBasis, QuadraticBasis, HermiteBasis]
)
def test_basis_functions(basis_function_class, delta=1e-6, rtol=1e-6):
    """
    Test evaluation and derivatives of the basis function class in the unit interval using finite-difference validation.

    Parameters
    ----------
    basis_function_class : class
        The class of the basis function to test (QuadraticBasis and HermiteBasis).
    delta : float, optional
        Step size for finite-difference approximation (default is 1e-6).
    rtol : float, optional
        Relative tolerance for derivative comparisons (default is 1e-6).

    Returns
    -----
    - Verifies that the shape of `eval`, `first_derivative`, and `second_derivative` outputs match expectations.
    - Compares the analytical first and second derivatives against central finite-difference approximations.
    """

    basis_function = basis_function_class()

    # evaluate quadrature and hermite basis functions plus their first and second derivatives in desired values
    x_i = np.array([0.1, 0.3, 0.7, 0.9])
    v_ni = basis_function.eval(x_i)
    d_ni = basis_function.first_derivative(x_i)
    # s_d_ni = basis_function.second_derivative(x_i)
    assert v_ni.shape == (basis_function.nb_nodes, len(x_i))
    assert v_ni.shape == d_ni.shape
    # assert d_ni.shape == s_d_ni.shape

    # Finite-differences test of the derivative
    v_plus_ni = basis_function.eval(x_i + delta)
    v_minus_ni = basis_function.eval(x_i - delta)
    d_fd_ni = (v_plus_ni - v_minus_ni) / (2 * delta)
    # ds_fd_ni = (v_plus_ni - 2 * v_ni + v_minus_ni) / (delta ** 2)

    # Compare analytic vs FD
    np.testing.assert_allclose(d_ni, d_fd_ni, rtol=rtol)
    # np.testing.assert_allclose(s_d_ni, ds_fd_ni, rtol=1e-03)


@pytest.mark.parametrize("basis_function_class", [QuadraticBasis, HermiteBasis])
def test_basis_function(basis_function_class, delta=1e-6, rtol=1e-6):
    """
    Test evaluation and derivatives of a basis function class in an arbitrary interval using finite-difference validation.

    Parameters
    ----------
    basis_function_class : class
        The class of the basis function to test (QuadraticBasis and HermiteBasis).
    delta : float, optional
        Step size for finite-difference approximation (default is 1e-6).
    rtol : float, optional
        Relative tolerance for derivative comparisons (default is 1e-6).
    Returns
    -----
    - Verifies that the shape of `eval`, `first_derivative`, and `second_derivative` outputs match expectations.
    - Compares the analytical first and second derivatives against central finite-difference approximations.
    """
    basis_function = basis_function_class()

    # Define mesh and evaluation points
    y_n = np.array([0.5, 1])
    y_k = np.linspace(0.5, 1, 11)

    for i in range(basis_function.nb_nodes):
        w_g = np.zeros(basis_function.nb_nodes)
        w_g[i] = 1.0

        v_ni = basis_function.interpolate(y_n, w_g, y_k)
        d_ni = basis_function.interpolate_first_derivative(y_n, w_g, y_k)
        sd_ni = basis_function.interpolate_second_derivative(y_n, w_g, y_k)

        assert v_ni.shape == (len(y_k),)
        assert d_ni.shape == (len(y_k),)
        assert sd_ni.shape == (len(y_k),)

        # Finite-differences test of the derivative
        v_plus_ni = basis_function.interpolate(y_n, w_g, y_k + delta)
        v_minus_ni = basis_function.interpolate(y_n, w_g, y_k - delta)
        d_fd_ni = (v_plus_ni - v_minus_ni) / (2 * delta)
        sd_fd_ni = (v_plus_ni - 2 * v_ni + v_minus_ni) / (delta**2)

        # Compare analytic vs FD
        np.testing.assert_allclose(d_ni, d_fd_ni, rtol=rtol, atol=1e-09)
        np.testing.assert_allclose(sd_ni, sd_fd_ni, rtol=rtol, atol=1e-2)


@pytest.mark.parametrize("basis_function_class", [QuadraticBasis, HermiteBasis])
def test_plot_basis_function(basis_function_class, plot=True):
    """
    Plot the basis functions and their derivatives in an arbitrary interval.

    Returns
    -----
    visualization of basis functions and their derivatives
    """
    basis_function = basis_function_class()
    y_n = np.array([0, 0.5])
    y_k = np.linspace(0, 0.5, 100)

    if plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

        for i in range(basis_function.nb_nodes):
            w_g = np.zeros(basis_function.nb_nodes)
            w_g[i] = 1.0
            v_ni = basis_function.interpolate(y_n, w_g, y_k)
            d_ni = basis_function.interpolate_first_derivative(y_n, w_g, y_k)
            sd_ni = basis_function.interpolate_second_derivative(y_n, w_g, y_k)

            axs[0].plot(y_k, v_ni, label=f"ϕ_{i}")
            axs[1].plot(y_k, d_ni, label=f"ϕ_{i}'")
            axs[2].plot(y_k, sd_ni, label=f"ϕ_{i}''")

        axs[0].set_title("Basis Functions ϕ_i(x)")
        axs[1].set_title("First Derivatives ϕ_i'(x)")
        axs[2].set_title("Second Derivatives ϕ_i''(x)")

        for ax in axs:
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()


@pytest.mark.parametrize("shape_function_class", [BilinearShapeFunctions])
def test_plot_shape_function(shape_function_class, plot=True):
    shape = shape_function_class()
    s_vals = np.linspace(-1, 1, 50)
    n_vals = np.linspace(-1, 1, 50)
    S, N = np.meshgrid(s_vals, n_vals)

    if plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(10, 8), subplot_kw={'projection': '3d'})
        titles = [r"$\hat{\varphi}_1$", r"$\hat{\varphi}_2$", r"$\hat{\varphi}_3$", r"$\hat{\varphi}_4$"]

        for i, ax in enumerate(axs.flat):
            Z = shape.eval(S, N)[i]
            ax.plot_surface(S, N, Z, cmap='viridis')
            ax.set_title(titles[i])
            ax.set_xlabel("ξ (s)")
            ax.set_ylabel("η (n)")
            ax.set_zlabel("φ")
        plt.tight_layout()
        plt.show()