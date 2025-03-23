import numpy as np
import pytest

from VascularFlow.Numerics.Assembly import assemble_system_matrix
from VascularFlow.Numerics.BasisFunctions import LinearBasis, QuadraticBasis
from VascularFlow.Numerics.ElementMatrices import first_first, second_second


def test_assemble_2x2():
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


@pytest.mark.parametrize("nb_elements", [1, 2, 3, 4, 5])
def test_assemble_rank_2x2(nb_elements):
    element_matrices_nn = first_first(1, LinearBasis())
    element_matrices_enn = element_matrices_nn.reshape((1, 2, 2)) * np.ones(
        nb_elements).reshape(
        (nb_elements, 1, 1)
    )
    system_matrix_gg = assemble_system_matrix(element_matrices_enn)
    assert system_matrix_gg.shape == (nb_elements + 1, nb_elements + 1)
    assert np.linalg.matrix_rank(system_matrix_gg) == nb_elements


def test_assemble_3x3():
    element_matrices_nn = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    element_matrices_enn = element_matrices_nn.reshape((1, 3, 3)) * np.ones(3).reshape(
        (3, 1, 1)
    )

    system_matrix_gg = assemble_system_matrix(element_matrices_enn)

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


@pytest.mark.parametrize("nb_elements", [1, 2, 3, 4, 5])
def test_assemble_rank_first_first_3x3(nb_elements):
    element_matrices_nn = first_first(2, QuadraticBasis())
    element_matrices_enn = element_matrices_nn.reshape((1, 3, 3)) * np.ones(
        nb_elements).reshape(
        (nb_elements, 1, 1)
    )
    system_matrix_gg = assemble_system_matrix(element_matrices_enn)
    assert system_matrix_gg.shape == (2 * nb_elements + 1, 2 * nb_elements + 1)
    assert np.linalg.matrix_rank(system_matrix_gg) == 2 * nb_elements


@pytest.mark.parametrize("nb_elements", [1, 2, 3, 4, 5])
def test_assemble_rank_second_second_3x3(nb_elements):
    element_matrices_nn = second_second(1, QuadraticBasis())
    element_matrices_enn = element_matrices_nn.reshape((1, 3, 3)) * np.ones(
        nb_elements).reshape(
        (nb_elements, 1, 1)
    )
    system_matrix_gg = assemble_system_matrix(element_matrices_enn)
    assert system_matrix_gg.shape == (2 * nb_elements + 1, 2 * nb_elements + 1)
    assert np.linalg.matrix_rank(system_matrix_gg) == nb_elements
