"""
Tests for evaluating matrices for a single element used in 1D finite element methods.
Tests for evaluating matrices for a single element used in 2D finite element methods.

This module verifies:
- Accuracy of the shape of the element matrices
- Accuracy of element matrices using manual calculations of the integrals in an arbitrary interval.

Basis function types:
1D:
    - LinearBasis
    - QuadraticBasis
    - HermiteBasis
2D:
    - Bi-linear shape functions
    - adini clough melosh (ACM)
"""

import numpy as np
import pytest

from VascularFlow.Numerics.BasisFunctions import (
    LinearBasis,
    QuadraticBasis,
    HermiteBasis,
    BilinearShapeFunctions,
)
from VascularFlow.Numerics.ElementMatrices import (
    stiffness_matrix_second_derivative,
    stiffness_matrix_first_derivative,
    stiffness_matrix_fourth_derivative,
    load_vector,
    mass_matrix_fourth_derivatives,
    element_matrices_or_vectors_2d,
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


######## Definition of matrices for a single reference quadrilateral element used in 2D finite element methods. ########


@pytest.mark.parametrize("shape_function_class", [BilinearShapeFunctions])
def test_element_matrix_2d(shape_function_class):
    """
    Test the element_matrix function for mass and stiffness matrix generation
    using a BilinearShapeFunctions instance at the center of the reference square.
    """

    # Create bilinear basis
    shape_function = shape_function_class()

    # Compute mass and stiffness element matrices
    M_elem = element_matrices_or_vectors_2d(
        shape_function, nb_quad_pts_2d=9, dx=0.05, kind="mass"
    )
    K_elem = element_matrices_or_vectors_2d(
        shape_function, nb_quad_pts_2d=9, dx=0.05, kind="stiffness"
    )
    # Example constant source term function f(ξ, η) = 1
    f_func = lambda s, n: 1.0   # just for testing
    F = element_matrices_or_vectors_2d(
        shape_function, nb_quad_pts_2d=9, dx=0.05, kind="source", f=f_func
    )

    #print("Mass matrix M:\n", M_elem)
    print("\nStiffness matrix K:\n", K_elem)
    #print("\nSource vector F:\n", F)
