import numpy as np

from VascularFlow.Numerics.BasisFunctions import LinearBasis
from VascularFlow.Numerics.ElementMatrices import first_first


def test_first_first():
    basis_function = LinearBasis()
    element_matrix = first_first(1, basis_function)

    # This is the Laplace matrix for linear elements
    np.testing.assert_allclose(element_matrix, [[1, -1], [-1, 1]])
