import numpy as np

from VascularFlow.Numerics.Assembly import assemble_system_matrix
from VascularFlow.Numerics.BasisFunctions import LinearBasis
from VascularFlow.Numerics.ElementMatrices import first_first


def test_assemble_system_matrix():
    element_matrices_nn = first_first(1, LinearBasis())

    np.testing.assert_allclose(element_matrices_nn, [[1, -1], [-1, 1]])

    element_matrices_enn = element_matrices_nn.reshape((1, 2, 2)) * np.ones(3).reshape(
        (3, 1, 1)
    )

    system_matrix_gg = assemble_system_matrix(element_matrices_enn)

    assert system_matrix_gg.shape == (4, 4)

    np.testing.assert_allclose(
        system_matrix_gg,
        [[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]],
        atol=1e-6,
    )
