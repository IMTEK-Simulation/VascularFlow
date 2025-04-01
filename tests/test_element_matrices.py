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
    "basis_function_class, element_matrix_or_vector",
    [
        (HermiteBasis, stiffness_matrix_fourth_derivative),
        (HermiteBasis, mass_matrix_fourth_derivatives),
        (HermiteBasis, load_vector),
        (QuadraticBasis, stiffness_matrix_first_derivative),
        (QuadraticBasis, load_vector),
        (LinearBasis, stiffness_matrix_second_derivative),
    ],
)
def test_element_matrix_or_vector(basis_function_class, element_matrix_or_vector):
    nb_quadrature_points = 3
    y_n = np.array([0, 0.5])
    basis_function = basis_function_class()
    result = element_matrix_or_vector(nb_quadrature_points, y_n, basis_function)
    print(result.shape)
    print(result)
    expected_lookup = {
        (HermiteBasis, stiffness_matrix_fourth_derivative): np.array(
            [
                [96, 48, -96, 48],
                [48, 32, -48, 16],
                [-96, -48, 96, -48],
                [48, 16, -48, 32],
            ]
        ),
        (HermiteBasis, mass_matrix_fourth_derivatives): np.array(
            [
                [156, 22, 54, -13],
                [22, 4, 13, -3],
                [54, 13, 156, -22],
                [-13, -3, -22, 4],
            ]
        )
        / 840,
        (HermiteBasis, load_vector): np.array([1 / 4, 1 / 24, 1 / 4, -1 / 24]),
        (QuadraticBasis, stiffness_matrix_first_derivative): np.array(
            [
                [-0.5, 0.66, -0.16],
                [-0.66, 0.0, 0.66],
                [0.16, -0.66, 0.5],
            ]
        ),
        (QuadraticBasis, load_vector): np.array([0.083, 0.33, 0.083]),
        (LinearBasis, stiffness_matrix_second_derivative): np.array(
            [
                [2, -2],
                [-2, 2],
            ]
        ),
    }

    # Tolerances for each case
    atol_lookup = {
        (HermiteBasis, stiffness_matrix_fourth_derivative): 1e-12,
        (HermiteBasis, mass_matrix_fourth_derivatives): 8e-4,
        (HermiteBasis, load_vector): 1e-8,
        (QuadraticBasis, stiffness_matrix_first_derivative): 7e-3,
        (QuadraticBasis, load_vector): 4e-3,
        (LinearBasis, stiffness_matrix_second_derivative): 1e-10,
    }

    expected = expected_lookup[(basis_function_class, element_matrix_or_vector)]
    atol = atol_lookup[(basis_function_class, element_matrix_or_vector)]

    # Assertions
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, atol=atol)
