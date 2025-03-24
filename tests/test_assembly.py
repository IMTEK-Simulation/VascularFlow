import numpy as np

from VascularFlow.Numerics.Assembly import assemble_system_matrix_1dof, assemble_system_matrix_2dof, assemble_force_matrix_2dof
from VascularFlow.Numerics.BasisFunctions import LinearBasis, HermiteBasis
from VascularFlow.Numerics.ElementMatrices import first_first, second_second, force_matrix, mass_matrix


def test_assemble_2x2():
    element_matrices_nn = first_first(1, LinearBasis())

    np.testing.assert_allclose(element_matrices_nn, [[1, -1], [-1, 1]])

    element_matrices_enn = element_matrices_nn.reshape((1, 2, 2)) * np.ones(3).reshape(
        (3, 1, 1)
    )

    system_matrix_gg = assemble_system_matrix_1dof(element_matrices_enn)

    assert system_matrix_gg.shape == (4, 4)

    np.testing.assert_allclose(
        system_matrix_gg,
        [[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]],
        atol=1e-6,
    )


def test_assemble_3x3():
    element_matrices_nn = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    element_matrices_enn = element_matrices_nn.reshape((1, 3, 3)) * np.ones(3).reshape(
        (3, 1, 1)
    )

    system_matrix_gg = assemble_system_matrix_1dof(element_matrices_enn)

    assert system_matrix_gg.shape == (7, 7)

    np.testing.assert_allclose(
        system_matrix_gg,
        [
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 2, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 2, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
        ],
        atol=1e-6,
    )


def test_assemble_hermite():
    element_matrices_nn = second_second(3, 2, HermiteBasis())
    element_matrices_enn = element_matrices_nn.reshape((1, 4, 4)) * np.ones(3).reshape(3, 1, 1)
    print(element_matrices_enn.shape)
    system_matrix_gg = assemble_system_matrix_2dof(element_matrices_enn)
    print(system_matrix_gg)

def test_assemble_mass_matrix():
    element_matrices_nn = mass_matrix(3, 1, HermiteBasis())
    element_matrices_enn = element_matrices_nn.reshape((1, 4, 4)) * np.ones(2).reshape(2, 1, 1)
    system_matrix_gg = assemble_system_matrix_2dof(element_matrices_enn)
    print(system_matrix_gg.shape)
    print(system_matrix_gg)



def test_assemble_force_matrix():
    element_matrices_nn = force_matrix(1)
    element_matrices_enn = element_matrices_nn.reshape((1, 1, 4)) * np.ones(3).reshape(3, 1, 1)
    system_matrix_gg = assemble_force_matrix_2dof(element_matrices_enn)
    print(system_matrix_gg)



