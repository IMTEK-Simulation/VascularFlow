import numpy as np

from VascularFlow.Numerics.BasisFunctions import LinearBasis, QuadraticBasis, HermiteBasis
from VascularFlow.Numerics.ElementMatrices import first_first, second_second, force_matrix, mass_matrix, dx_matrix_mass

def test_first_first():
    basis_function = LinearBasis()
    element_matrix = first_first(1, basis_function)
    # This is the Laplace matrix for linear elements
    np.testing.assert_allclose(element_matrix, [[1, -1], [-1, 1]])


def test_second_second():
    basis_function = HermiteBasis()
    element_matrix = second_second(3, 1, basis_function)
    assert element_matrix.shape == (4, 4)
    print(element_matrix)


def test_force_matrix():
    dx = 1
    print(force_matrix(dx))

def test_dx_matrix_mass():
    dx = 1
    print(dx_matrix_mass(dx))


def test_mass_matrix():
    basis_function = HermiteBasis()
    element_matrix = mass_matrix(3, 2, basis_function)
    print(element_matrix.shape)
    print(element_matrix)