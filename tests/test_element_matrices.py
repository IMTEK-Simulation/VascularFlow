"""
Tests for evaluating matrices for a single element used in 1D finite element methods.

This module verifies:
- Accuracy of the shape of the element matrices
- Accuracy of element matrices using manual calculations of the integrals in an arbitrary interval.

Tested basis types:
- LinearBasis
- QuadraticBasis
- HermiteBasis
"""

import numpy as np
import pytest

from VascularFlow.Numerics.BasisFunctions import (
    LinearBasis,
    QuadraticBasis,
    HermiteBasis,
)
from VascularFlow.Numerics.ElementMatrices import (
    stiffness_matrix_second_derivative,
    stiffness_matrix_first_derivative,
    stiffness_matrix_fourth_derivative,
    load_vector,
    mass_matrix_fourth_derivatives,
)


@pytest.mark.parametrize(
    "basis_class, expected, atol",
    [
        (
            HermiteBasis,
            np.array(
                [
                    [96, 48, -96, 48],
                    [48, 32, -48, 16],
                    [-96, -48, 96, -48],
                    [48, 16, -48, 32],
                ]
            ),
            1e-12,
        )
    ],
)
def test_stiffness_matrix_fourth_derivative(basis_class, expected, atol):
    y_n = np.array([0, 0.5])
    k = stiffness_matrix_fourth_derivative(3, y_n, basis_class())
    assert k.shape == expected.shape
    np.testing.assert_allclose(k, expected, atol=atol)


@pytest.mark.parametrize(
    "basis_class, expected, atol",
    [
        (
            HermiteBasis,
            np.array(
                [
                    [156, 22, 54, -13],
                    [22, 4, 13, -3],
                    [54, 13, 156, -22],
                    [-13, -3, -22, 4],
                ]
            )
            / 840,
            8e-4,
        )
    ],
)
def test_mass_matrix_fourth_derivatives(basis_class, expected, atol):
    y_n = np.array([0, 0.5])
    m = mass_matrix_fourth_derivatives(3, y_n, basis_class())
    assert m.shape == expected.shape
    np.testing.assert_allclose(m, expected, atol=atol)


@pytest.mark.parametrize(
    "basis_class, expected, atol",
    [(HermiteBasis, np.array([1 / 4, 1 / 24, 1 / 4, -1 / 24]), 1e-8)],
)
def test_load_vector_fourth_derivatives(basis_class, expected, atol):
    y_n = np.array([0, 0.5])
    f = load_vector(3, y_n, basis_class())
    assert f.shape == expected.shape
    np.testing.assert_allclose(f, expected, atol=atol)


@pytest.mark.parametrize(
    "basis_class, expected, atol",
    [
        (
            QuadraticBasis,
            np.array([[-0.5, 0.66, -0.16], [-0.66, 0.0, 0.66], [0.16, -0.66, 0.5]]),
            7e-3,
        )
    ],
)
def test_stiffness_matrix_first_derivative(basis_class, expected, atol):
    y_n = np.array([0, 0.5])
    k = stiffness_matrix_first_derivative(4, y_n, basis_class())
    assert k.shape == expected.shape
    np.testing.assert_allclose(k, expected, atol=atol)


@pytest.mark.parametrize(
    "basis_class, expected, atol",
    [(QuadraticBasis, np.array([0.083, 0.33, 0.083]), 4e-3)],
)
def test_load_vector_first_derivative(basis_class, expected, atol):
    y_n = np.array([0, 0.5])
    f = load_vector(3, y_n, basis_class())
    assert f.shape == expected.shape
    np.testing.assert_allclose(f, expected, atol=atol)


@pytest.mark.parametrize(
    "basis_class, expected, atol",
    [
        (
            LinearBasis,
            np.array(
                [
                    [4, -4],
                    [-4, 4],
                ]
            ),
            1e-10,
        )
    ],
)
def test_stiffness_matrix_second_derivative(basis_class, expected, atol):
    y_n = np.array([0, 0.25])
    k = stiffness_matrix_second_derivative(3, y_n, basis_class())
    assert k.shape == expected.shape
    np.testing.assert_allclose(k, expected, atol=atol)
