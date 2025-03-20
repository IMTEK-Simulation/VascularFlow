import numpy as np

from VascularFlow.Numerics.BasisFunctions import LinearBasis, QuadraticBasis
from VascularFlow.Numerics.ElementMatrices import first_first, second_second


def test_first_first():
    basis_function = LinearBasis()
    element_matrix = first_first(1, basis_function)

    # This is the Laplace matrix for linear elements
    np.testing.assert_allclose(element_matrix, [[1, -1], [-1, 1]])


def test_second_second():
    basis_function = QuadraticBasis()
    element_matrix = second_second(1, basis_function)

    # This is the Laplace matrix for linear elements
    np.testing.assert_allclose(
        element_matrix, [[16, -32, 8], [-32, 64, -16], [8, -16, 4]]
    )
