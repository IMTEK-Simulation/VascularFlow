import numpy as np

from VascularFlow.Numerics.BasisFunctions import (
    LinearBasis,
    QuadraticBasis,
    HermiteBasis,
)
from VascularFlow.Numerics.ElementMatrices import (
    stiffness_matrix_second_derivative,
    stiffness_matrix_first_derivative,
    stiffness_matrix_fourth_derivative,
    load_vector_fourth_derivatives,
    load_vector_first_derivative,
    mass_matrix_fourth_derivatives,
)


def test_stiffness_matrix_fourth_derivative():
    basis_function = HermiteBasis()
    element_matrix = stiffness_matrix_fourth_derivative(3, 1, basis_function)
    assert element_matrix.shape == (4, 4)
    print(element_matrix)


def test_mass_matrix_fourth_derivatives():
    basis_function = HermiteBasis()
    element_matrix = mass_matrix_fourth_derivatives(3, 1, basis_function)
    assert element_matrix.shape == (4, 4)
    print(element_matrix)


def test_load_vector_fourth_derivatives():
    dx_e = 1
    element_load_vector = load_vector_fourth_derivatives(dx_e)
    assert element_load_vector.shape == (4,)
    print(element_load_vector)


def test_stiffness_matrix_first_derivative():
    basis_function = LinearBasis()
    element_matrix = stiffness_matrix_first_derivative(1, basis_function)
    assert element_matrix.shape == (2, 2)
    print(element_matrix)


def test_load_vector_first_derivative():
    dx_e = 1
    element_load_vector = load_vector_first_derivative(dx_e)
    assert element_load_vector.shape == (2,)
    print(element_load_vector)


def test_stiffness_matrix_second_derivative():
    basis_function = LinearBasis()
    element_matrix = stiffness_matrix_second_derivative(1, basis_function)
    # This is the Laplace matrix for linear elements
    np.testing.assert_allclose(element_matrix, [[1, -1], [-1, 1]])
